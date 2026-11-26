Studies with cell models on the scene of `../lobule_toy` (the same mesh and `zonation` map), run
end to end by the tests of `crates/slidesim`:

- `cells.study.json`: the linear model `../../models/linear_uptake.xml` (uptake of `A_ext` from
  the blood into the cell at a rate from the zonation, no decay), coupled to the field
  `substrate` which the flow carries from the portal wall (1 mM) to the central wall. The amount
  in the blood and the cells changes by the flows through the walls only.
- `spt.study.json`: the SPT cell model `../../models/spt_liver_cell.xml` with pattern 7 (the
  protein set from outside, here a table over the zonation, high at the central end), `S_ext`
  and `P_ext` coupled to the fields `S` (5 mM at the portal wall) and `P`, for 60 min. The
  threshold of the toxic product `T` for necrosis is 1.5 mM instead of the model's 5 mM, so that
  necrosis starts within the hour, in the cells of high protein.
