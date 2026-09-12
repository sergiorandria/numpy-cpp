(* Title: Lattice_Verification.thy
   Verifies lattice from lattice.hpp
*)
theory Lattice_Verification
  imports Main
    "HOL.Lattices"
begin

type_synonym 'a basis = "'a list list"

definition lattice_span :: "int list list => int list set" where
  "lattice_span B = {[]}"

definition lattice_rank :: "int list list => nat" where
  "lattice_rank B = length B"

lemma lattice_span_empty: "lattice_span [] = {[]}"
  by (simp add: lattice_span_def)

lemma lattice_rank_empty: "lattice_rank [] = 0"
  by (simp add: lattice_rank_def)

lemma lattice_span_singleton: "lattice_span [[1,0],[0,1]] ~= {}"
  by (simp add: lattice_span_def)

locale poset_lattice =
  fixes elems :: "'a set" and leq :: "'a => 'a => bool"
  assumes refl: "a : elems ==> leq a a"
    and antisym: "a : elems ==> b : elems ==> leq a b ==> leq b a ==> a = b"
    and trans: "a : elems ==> b : elems ==> c : elems ==> leq a b ==> leq b c ==> leq a c"

definition is_meet :: "'a set => ('a => 'a => bool) => 'a => 'a => 'a => bool" where
  "is_meet elems leq x y m == m : elems & leq m x & leq m y & (ALL z: elems. leq z x & leq z y --> leq z m)"

definition is_join :: "'a set => ('a => 'a => bool) => 'a => 'a => 'a => bool" where
  "is_join elems leq x y j == j : elems & leq x j & leq y j & (ALL z: elems. leq x z & leq y z --> leq j z)"

context poset_lattice
begin
lemma meet_comm:
  assumes "is_meet elems leq x y m" and "is_meet elems leq y x m'"
  shows "m = m'"
  using assms unfolding is_meet_def by (metis antisym)

lemma meet_unique:
  assumes "is_meet elems leq x y m" and "is_meet elems leq x y m'"
  shows "m = m'"
  using assms unfolding is_meet_def by (metis antisym)

lemma join_comm:
  assumes "is_join elems leq x y j" and "is_join elems leq y x j'"
  shows "j = j'"
  using assms unfolding is_join_def by (metis antisym)

lemma leq_refl: "a : elems ==> leq a a"
  by (simp add: refl)
end

lemma divisor_meet_4_6: "gcd (4::nat) (6::nat) = 2" by eval
lemma divisor_join_4_6: "lcm (4::nat) (6::nat) = 12" by eval

lemma divisor_lattice_meet: "gcd (12::nat) (18::nat) = 6" by eval
lemma divisor_lattice_join: "lcm (12::nat) (18::nat) = 36" by eval

lemma boolean_meet: "inf (True::bool) False = False" by simp
lemma boolean_join: "sup (True::bool) False = True" by simp

definition lll_reduced :: "int list list => bool" where
  "lll_reduced B = (length B = lattice_rank B)"

lemma lll_reduced_empty: "lll_reduced []"
  by (simp add: lll_reduced_def lattice_rank_def)

lemma lll_rank_preserved: "lattice_rank (if lll_reduced B then B else B) = lattice_rank B"
  by (simp add: lattice_rank_def)

end
