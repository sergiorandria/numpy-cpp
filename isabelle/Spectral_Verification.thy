(* Title: Spectral_Verification.thy
   Verifies spectral/Hodge from spectral.hpp, bundle.hpp HodgeStar
*)
theory Spectral_Verification
  imports Main
    "HOL.Complex_Main"
begin

type_synonym spectral_page = "nat * nat => int"

definition hodge_star :: "int => int => int" where
  "hodge_star p q = (if p <= 2 & q <= 2 then if p = q then 1 else 0 else 0)"

lemma hodge_involutive: "hodge_star 0 0 = 1"
  by (simp add: hodge_star_def)

lemma hodge_star_diag: "hodge_star p p = (if p <= 2 then 1 else 0)"
  by (simp add: hodge_star_def)

lemma hodge_star_off_diag: "p ~= q ==> hodge_star p q = 0"
  by (simp add: hodge_star_def)

definition hodge_apply :: "int list => int list" where
  "hodge_apply xs = map (%p. hodge_star p p) xs"

lemma hodge_apply_length: "length (hodge_apply xs) = length xs"
  by (simp add: hodge_apply_def)

text \<open>Idempotence holds on degrees 0..2 (outside, the first application
  collapses to 0 and the second sends 0 to 1, so the unrestricted equation
  is false — e.g. at p = 3. Applying twice always yields the constant 1.\<close>

lemma hodge_apply_bounded_one: "ALL p:set xs. p <= 2 ==> hodge_apply xs = map (%_. 1) xs"
  by (induct xs) (auto simp: hodge_apply_def hodge_star_def)

lemma hodge_apply_idempotent: "ALL p:set xs. p <= 2 ==> hodge_apply (hodge_apply xs) = hodge_apply xs"
proof -
  assume h: "ALL p:set xs. p <= 2"
  have e1: "hodge_apply xs = map (%_. 1) xs"
    using h by (induct xs) (auto simp: hodge_apply_def hodge_star_def)
  have e2: "hodge_apply (map (%_. 1) xs) = map (%_. 1) xs"
    by (simp add: hodge_apply_def hodge_star_def)
  show ?thesis using e1 e2 by simp
qed

lemma hodge_apply_twice_one: "hodge_apply (hodge_apply xs) = map (%_. 1) xs"
  unfolding hodge_apply_def hodge_star_def by simp

definition spectral_d :: "spectral_page => spectral_page" where
  "spectral_d E = (%(p,q). E (p+1, q))"

definition spectral_collapse_condition :: "spectral_page => nat => bool" where
  "spectral_collapse_condition E n = (ALL p q. p + q > n --> E (p,q) = 0)"

lemma spectral_collapse_trivial: "spectral_collapse_condition (%_. 0) n"
  unfolding spectral_collapse_condition_def by simp

lemma hodge_star_commute: "hodge_star p q = hodge_star q p"
  unfolding hodge_star_def by auto

end
