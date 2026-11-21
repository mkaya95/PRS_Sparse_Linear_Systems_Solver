# Our solvers against Eigen, after the optimizations

`solve.csv`: one run of each solver on six matrices, tolerance 1e-8, right-hand side b = A·x_true with a random
x_true, so the correct answer is known. Run on 2026-11-21 at commit 4b1e51c with

```bash
python3 scripts/run_bench.py --build build/release --solvers eigen-E1,ours-jacobi,ours-sell-jacobi,ours-sgs,eigen-E4,ours-ic,ours-ic-fixed,ours-ic0,ours-rcm-ic0 \
    --data-dir ~/p08/data --matrices ecology2,apache2,tmt_sym,G3_circuit,thermal2,hood --repeat 1 --warmup 0 --pin 0 \
    --out solve.csv --solutions solutions
```

The solution files themselves (10 MB) are not kept; the table has what they showed. "Difference" is the
relative difference ‖x₁ − x₂‖ / ‖x₂‖ between two solvers' answers; "error" is the same against x_true.

| Matrix | Iterations E1 / ours-jacobi | Difference | Iterations E4 / ours-ic | Difference | Error E1 / ours-jacobi | Error E4 / ours-ic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ecology2 | 2262 / 2262 | 9.7e-14 | 672 / 672 | 5.5e-7 | 1.4e-3 / 1.4e-3 | 1.4e-3 / 1.4e-3 |
| apache2 | 1881 / 1881 | 5.8e-14 | 504 / 504 | 1.1e-9 | 7.2e-5 / 7.2e-5 | 1.1e-4 / 1.1e-4 |
| tmt_sym | 1571 / 1568 | 1.0e-6 | 579 / 580 | 1.8e-8 | 1.9e-3 / 1.9e-3 | 1.6e-3 / 1.6e-3 |
| G3_circuit | 1015 / 1015 | 9.2e-14 | 251 / 251 | 1.9e-7 | 2.3e-4 / 2.3e-4 | 3.0e-4 / 3.0e-4 |
| thermal2 | 2346 / 2348 | 3.0e-7 | 1133 / 1524 | 7.3e-4 | 5.4e-4 / 5.4e-4 | 8.8e-4 / 1.8e-4 |
| hood | 2618 / 2627 | 6.0e-7 | 717 / 717 | 3.6e-6 | 3.9e-5 / 3.9e-5 | 2.8e-5 / 2.8e-5 |

Jacobi and incomplete Cholesky give Eigen's answer to 5 to 13 digits, and are exactly as far from x_true as Eigen,
on every matrix but thermal2 for IC, explained below. The differences from Eigen are always smaller than the distance
from x_true (at least 8 times, on hood; usually by thousands): CG stops at a residual of 1e-8, which on these
matrices leaves an error of 1e-5 to 3e-3 in x. Bit-for-bit equality is not possible, since Eigen sums in a different order.

## thermal2: ours-ic 1524 iterations, Eigen E4 1133

Not a bug. The factors are the same: compared column by column on thermal2, Eigen's L and ours have the same
pattern in all 1,228,045 columns and values within 1e-13 relative (rounding). The difference is where CG stops.
From iteration 1000 the residual hovers just above 1e-8: at Eigen's last iteration, 1133, ours was 1.05e-8,
5 % above the line; it then rose again (up to 1.6e-8), came within 0.4 % of the line, and crossed only at 1524. Moving the line by 1 to 2 % moves both:

| Tolerance | Eigen E4 | ours-ic |
| --- | ---: | ---: |
| 1.02e-8 | 1110 | 1135 |
| 1.00e-8 | 1133 | 1524 |
| 0.99e-8 | 1522 | 1525 |

(`build/profile/methods/eigen_e4` and `cg_ic` with `--tol`.) Running longer made ours-ic closer to x_true than
Eigen (error 1.8e-4 against 8.8e-4), since the error keeps falling while the residual stalls.
