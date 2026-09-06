(* Title: Dual_Verification.thy
   Verifies Dual numbers from differential.hpp
*)
theory Dual_Verification
  imports Main
    "HOL.Complex_Main"
begin

record 'a dual = val :: 'a  dval :: 'a

definition dual_add :: "'a::comm_ring_1 dual => 'a dual => 'a dual" where
  "dual_add a b = (| val = val a + val b, dval = dval a + dval b |)"

definition dual_mul :: "'a::comm_ring_1 dual => 'a dual => 'a dual" where
  "dual_mul a b = (| val = val a * val b, dval = val a * dval b + dval a * val b |)"

definition dual_sin :: "real dual => real dual" where
  "dual_sin a = (| val = sin (val a), dval = cos (val a) * dval a |)"

definition dual_cos :: "real dual => real dual" where
  "dual_cos a = (| val = cos (val a), dval = - sin (val a) * dval a |)"

definition dual_exp :: "real dual => real dual" where
  "dual_exp a = (| val = exp (val a), dval = exp (val a) * dval a |)"

lemma dual_add_val: "val (dual_add a b) = val a + val b"
  by (simp add: dual_add_def)

lemma dual_add_dval: "dval (dual_add a b) = dval a + dval b"
  by (simp add: dual_add_def)

lemma dual_add_comm: "dual_add a b = dual_add b a"
  by (simp add: dual_add_def add.commute)

lemma dual_add_assoc: "dual_add (dual_add a b) c = dual_add a (dual_add b c)"
  by (simp add: dual_add_def)

lemma dual_mul_val: "val (dual_mul a b) = val a * val b"
  by (simp add: dual_mul_def)

lemma dual_mul_dval: "dval (dual_mul a b) = val a * dval b + dval a * val b"
  by (simp add: dual_mul_def)

lemma dual_mul_comm: "dual_mul a b = dual_mul b a"
  unfolding dual_mul_def by (simp add: mult.commute add.commute)

lemma dual_mul_assoc: "val (dual_mul (dual_mul a b) c) = val (dual_mul a (dual_mul b c))"
  by (simp add: dual_mul_def mult.assoc)

lemma dual_mul_constexpr:
  fixes a :: "real dual" and b :: "real dual"
  assumes "a = (| val = 2, dval = 1 |)" "b = (| val = 3, dval = 0 |)"
  shows "dual_mul a b = (| val = 6, dval = 3 |)"
  using assms by (simp add: dual_mul_def)

lemma dual_sin_val: "val (dual_sin a) = sin (val a)"
  by (simp add: dual_sin_def)

lemma dual_sin_dval: "dval (dual_sin a) = cos (val a) * dval a"
  by (simp add: dual_sin_def)

lemma dual_cos_val: "val (dual_cos a) = cos (val a)"
  by (simp add: dual_cos_def)

lemma dual_chain_sin: "dval (dual_sin (| val = x, dval = 1 |)) = cos x"
  by (simp add: dual_sin_def)

lemma dual_chain_exp: "dval (dual_exp (| val = x, dval = 1 |)) = exp x"
  by (simp add: dual_exp_def)

lemma dual_exp_val: "val (dual_exp a) = exp (val a)"
  by (simp add: dual_exp_def)

end
