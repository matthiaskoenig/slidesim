// Smoke test of the wasm32-wasip1 static libraries: PETSc (Vec, Mat, KSP) and libCEED.
// Exits nonzero if any check fails.
#include <ceed.h>
#include <math.h>
#include <petscksp.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void check(const char *what, double got, double want, double tol)
{
  int ok = fabs(got - want) <= tol;
  printf("%-36s got %.15g want %.15g %s\n", what, got, want, ok ? "ok" : "FAIL");
  if (!ok) failures++;
}

// PETSc: ||ones(100)||_2 and a 1D Laplacian solve with GMRES + ILU whose exact solution is ones.
static PetscErrorCode petsc_smoke(void)
{
  const PetscInt n = 100;
  Vec            x, b, u;
  Mat            A;
  KSP            ksp;
  PetscReal      nrm, err;
  PetscInt       its;

  PetscFunctionBeginUser;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, &x));
  PetscCall(VecSet(x, 1.0));
  PetscCall(VecNorm(x, NORM_2, &nrm));
  check("petsc VecNorm(ones(100))", nrm, 10.0, 1e-14);

  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, n, n, 3, NULL, &A));
  for (PetscInt i = 0; i < n; i++) {
    if (i > 0) PetscCall(MatSetValue(A, i, i - 1, -1.0, INSERT_VALUES));
    PetscCall(MatSetValue(A, i, i, 2.0, INSERT_VALUES));
    if (i < n - 1) PetscCall(MatSetValue(A, i, i + 1, -1.0, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(VecDuplicate(x, &b));
  PetscCall(VecDuplicate(x, &u));
  PetscCall(MatMult(A, x, b));

  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetTolerances(ksp, 1e-12, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, b, u));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(VecAXPY(u, -1.0, x));
  PetscCall(VecNorm(u, NORM_INFINITY, &err));
  printf("petsc KSPSolve iterations             %" PetscInt_FMT "\n", its);
  check("petsc KSPSolve max error", err, 0.0, 1e-9);

  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// libCEED backend, always the full resource string: the /cpu/self prefix resolves to different
// backends natively (avx/blocked with -march=native) and in wasm (opt/blocked), see
// scripts/deps/patches/README.md.
#define CEED_RESOURCE "/cpu/self/opt/blocked"

// libCEED: CeedVectorNorm and the gallery mass operator on [0, 1] with linear elements:
// sum(M * ones) is the length of the domain.
static int ceed_smoke(void)
{
  enum { nelem = 8, P = 2, Q = 2, nnode = nelem + 1 };
  Ceed                ceed;
  const char         *resource;
  CeedVector          v, X, U, V, qdata;
  CeedElemRestriction r, rq;
  CeedBasis           basis;
  CeedQFunction       qf_build, qf_apply;
  CeedOperator        op_build, op_apply;
  CeedScalar          nrm, coords[nnode], sum = 0.0;
  CeedInt             offsets[nelem * P];
  const CeedScalar   *vv;

  if (CeedInit(CEED_RESOURCE, &ceed)) return 1;
  CeedGetResource(ceed, &resource);
  printf("ceed resource                        %s\n", resource);
  if (strcmp(resource, CEED_RESOURCE)) {
    printf("ceed resource is not %s FAIL\n", CEED_RESOURCE);
    failures++;
  }

  CeedVectorCreate(ceed, 100, &v);
  CeedVectorSetValue(v, 2.0);
  CeedVectorNorm(v, CEED_NORM_2, &nrm);
  check("ceed CeedVectorNorm(2*ones(100))", nrm, 20.0, 1e-14);
  CeedVectorDestroy(&v);

  for (CeedInt i = 0; i < nnode; i++) coords[i] = (CeedScalar)i / nelem;
  for (CeedInt e = 0; e < nelem; e++) offsets[e * P] = e, offsets[e * P + 1] = e + 1;
  CeedVectorCreate(ceed, nnode, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_COPY_VALUES, coords);
  CeedVectorCreate(ceed, nnode, &U);
  CeedVectorSetValue(U, 1.0);
  CeedVectorCreate(ceed, nnode, &V);
  CeedVectorCreate(ceed, nelem * Q, &qdata);

  CeedElemRestrictionCreate(ceed, nelem, P, 1, 1, nnode, CEED_MEM_HOST, CEED_COPY_VALUES, offsets, &r);
  CeedElemRestrictionCreateStrided(ceed, nelem, Q, 1, nelem * Q, CEED_STRIDES_BACKEND, &rq);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, P, Q, CEED_GAUSS, &basis);

  CeedQFunctionCreateInteriorByName(ceed, "Mass1DBuild", &qf_build);
  CeedOperatorCreate(ceed, qf_build, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_build);
  CeedOperatorSetField(op_build, "dx", r, basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_build, "weights", CEED_ELEMRESTRICTION_NONE, basis, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_build, "qdata", rq, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
  CeedOperatorApply(op_build, X, qdata, CEED_REQUEST_IMMEDIATE);

  CeedQFunctionCreateInteriorByName(ceed, "MassApply", &qf_apply);
  CeedOperatorCreate(ceed, qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_apply);
  CeedOperatorSetField(op_apply, "u", r, basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "qdata", rq, CEED_BASIS_NONE, qdata);
  CeedOperatorSetField(op_apply, "v", r, basis, CEED_VECTOR_ACTIVE);
  CeedOperatorApply(op_apply, U, V, CEED_REQUEST_IMMEDIATE);

  CeedVectorGetArrayRead(V, CEED_MEM_HOST, &vv);
  for (CeedInt i = 0; i < nnode; i++) sum += vv[i];
  CeedVectorRestoreArrayRead(V, &vv);
  check("ceed sum(M * ones) on [0, 1]", sum, 1.0, 1e-14);

  CeedOperatorDestroy(&op_apply);
  CeedOperatorDestroy(&op_build);
  CeedQFunctionDestroy(&qf_apply);
  CeedQFunctionDestroy(&qf_build);
  CeedBasisDestroy(&basis);
  CeedElemRestrictionDestroy(&rq);
  CeedElemRestrictionDestroy(&r);
  CeedVectorDestroy(&qdata);
  CeedVectorDestroy(&V);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&X);
  return CeedDestroy(&ceed);
}

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(petsc_smoke());
  PetscCall(PetscFinalize());
  if (ceed_smoke()) failures++;
  printf("%s\n", failures ? "SMOKE FAILED" : "SMOKE OK");
  return failures ? 1 : 0;
}
