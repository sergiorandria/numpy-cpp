# Isabelle/HOL Verification for numpy-cpp

This directory contains machine-checked proofs for the C++ implementation
using **Isabelle2025-2** (`HOL` + `HOL-Analysis`).

## Session

```
session NumpyCpp = HOL +
  theories Dual_Verification
           Differential_Verification
           Lattice_Verification
           Padic_Verification
```

Build with:

```bash
isabelle build -D isabelle -v
# or soft check
isabelle build -D isabelle -n -v
```

Expected: `Finished NumpyCpp (0:00:07)` with `100%` for all four theories
(as verified on `shishiki` with `polyml-5.9.2`).

## Correspondence

| Isabelle theory | C++ header | What is verified |
|---|---|---|
| `Dual_Verification.thy` | `include/np/differential.hpp:91` `Dual<T>` | Dual numbers `+`, `*`, `sin`, `cos`, `exp`; `dval` equals analytic derivative at `dval=1`; `2*3` constexpr check `kernel::check_dual_constexpr` |
| `Differential_Verification.thy` | `differential.hpp` `exterior_derivative`, `kernel::exterior_scalar`, `wedge`, `kernel::symbolic`/`simplify`, `hessian` | `exterior_derivative` dim, `wedge` antisymmetry `a∧b = -b∧a`, `d(d f)=0` by Schwarz (`∂²f/∂xᵢ∂xⱼ = ∂²f/∂xⱼ∂xᵢ`), symbolic `SAdd`/`SMul`/`SSin`/`SCos` differentiation correctness, `simplify` identities `0+x`, `1*x`, `0*x`, Hessian symmetry |
| `Lattice_Verification.thy` | `include/np/lattice.hpp:143` `Lattice<T>`, `PosetLattice` | `poset_lattice` locale (refl/antisym/trans), `is_meet`/`is_join` commutativity, Boolean lattice (`boolean_lattice 2`) `meet 1∧2=0` `join 1∨2=3` `is_lattice`/`distributive`, divisor lattice `gcd`/`lcm`, integer lattice `meet`=`dual(join(dual))`/`join`=span laws, `dual` involutive, LLL size-reduced/Lovász/volume preservation (axiomatized, matches `LLLStrategy`), `gram` symmetric, factory/strategy/visitor/observer/decorator/builder correspondence |
| `Padic_Verification.thy` | `include/np/padic.hpp:135` `Padic`, `PadicLattice`, `Hensel` | `padic_valuation` `25→2` `7→0`, `padic_norm` `25→1/25`, `is_padic_unit`, Hensel lift `2*3` unit, `PadicLattice`/`PadicDifferential` integration |

## Design patterns and modern C++20 mapped to HOL

C++ patterns (`Factory`, `Strategy`, `Visitor`, `Observer`, `Decorator`, `Prototype`,
`Builder`, `Template Method`) are modelled as HOL locales/records and shown to preserve
invariants (`rank`, `volume`, `is_lattice`). Modern C++20 features
(`concepts`, `std::span`, `std::ranges`, `std::variant`/`std::visit`, `constexpr`,
`std::shared_mutex`, `std::optional`) correspond to HOL `locale`, `set`, `variant`,
`consteval` (`check_dual_constexpr`).

## CI

Add to CI:

```yaml
- name: Isabelle verification
  run: isabelle build -D isabelle -v
```

Requires `isabelle` in `PATH` (`/opt/isabelle/bin/isabelle` on this machine).

## References

- Bott–Tu, *Differential Forms*; Spivak, *Calculus on Manifolds* (for `differential.hpp`)
- Micciancio–Goldwasser, *Complexity of Lattice Problems*; Lenstra–Lenstra–Lovász 1982
  (for `lattice.hpp`)
- Gouvea *p-adic Numbers*, Koblitz *p-adic Analysis*, Serre *Local Fields* (for `padic.hpp`)
- Isabelle/HOL: `HOL.Lattices`, `HOL.Complete_Lattices`, `HOL.Deriv`, `HOL.Transcendental`,
  `HOL.Complex_Main`, `HOL.Finite_Set`, `HOL.Number_Theory.Cong`
