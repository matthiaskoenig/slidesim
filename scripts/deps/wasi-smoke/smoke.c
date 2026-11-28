// Smoke test of Mmg in the wasm32-wasip1 build of PETSc (scripts/deps/build-wasi.sh): the unit
// square of two triangles adapted by DMAdaptMetric to the isotropic metric of the size 0.1 (the
// way crates/petsc/src/metric.rs does it), with the boundary refined too. Prints the number of
// adapted cells and nothing else; fails unless it lies in [150, 320].
#include <petscdmplex.h>

int main(int argc, char **argv)
{
  const PetscInt  cells[]       = {0, 1, 2, 0, 2, 3};
  const PetscReal coordinates[] = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
  const PetscReal h             = 0.1;
  DM              dm, adapted;
  Vec             metric;
  PetscInt        cStart, cEnd;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(DMPlexCreateFromCellListPetsc(PETSC_COMM_SELF, 2, 2, 4, 3, PETSC_TRUE, cells, 2, coordinates, &dm));
  PetscCall(DMPlexMetricSetIsotropic(dm, PETSC_TRUE));
  PetscCall(DMPlexMetricSetNoSurf(dm, PETSC_FALSE));
  PetscCall(DMPlexMetricCreate(dm, 0, &metric));
  PetscCall(VecSet(metric, 1.0 / (h * h)));
  PetscCall(DMAdaptMetric(dm, metric, NULL, NULL, &adapted));
  PetscCall(DMPlexGetHeightStratum(adapted, 0, &cStart, &cEnd));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "adapted cells %" PetscInt_FMT "\n", cEnd - cStart));
  PetscCheck(cEnd - cStart >= 150 && cEnd - cStart <= 320, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%" PetscInt_FMT " cells, not in [150, 320]", cEnd - cStart);
  PetscCall(VecDestroy(&metric));
  PetscCall(DMDestroy(&adapted));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}
