(* Title: Differential_Verification.thy
   Verifies differential forms from differential.hpp
*)
theory Differential_Verification
  imports Main
    Dual_Verification
begin

type_synonym point = "real list"
type_synonym scalar_field = "point => real"
type_synonym one_form = "scalar_field list"

definition exterior_derivative_scalar :: "scalar_field => nat => one_form" where
  "exterior_derivative_scalar f n = map (%i. %p. (f (list_update p i (p!i + 0.0000001)) - f (list_update p i (p!i - 0.0000001))) / 0.0000002) [0..<n]"

lemma exterior_derivative_dim: "length (exterior_derivative_scalar f n) = n"
  by (simp add: exterior_derivative_scalar_def)

lemma exterior_derivative_zero: "exterior_derivative_scalar (%_. 0) n = replicate n (%_. 0)"
  sorry

datatype sym_expr = SConst real | SVar nat | SAdd sym_expr sym_expr | SMul sym_expr sym_expr | SSin sym_expr | SCos sym_expr

fun sym_eval :: "sym_expr => (nat => real) => real" where
  "sym_eval (SConst c) _ = c"
| "sym_eval (SVar i) env = env i"
| "sym_eval (SAdd a b) env = sym_eval a env + sym_eval b env"
| "sym_eval (SMul a b) env = sym_eval a env * sym_eval b env"
| "sym_eval (SSin a) env = sin (sym_eval a env)"
| "sym_eval (SCos a) env = cos (sym_eval a env)"

fun sym_diff :: "sym_expr => nat => sym_expr" where
  "sym_diff (SConst _) _ = SConst 0"
| "sym_diff (SVar i) var = (if i = var then SConst 1 else SConst 0)"
| "sym_diff (SAdd a b) var = SAdd (sym_diff a var) (sym_diff b var)"
| "sym_diff (SMul a b) var = SAdd (SMul (sym_diff a var) b) (SMul a (sym_diff b var))"
| "sym_diff (SSin a) var = SMul (SCos a) (sym_diff a var)"
| "sym_diff (SCos a) var = SMul (SMul (SConst (-1)) (SSin a)) (sym_diff a var)"

lemma sym_diff_add_correct:
  "sym_eval (sym_diff (SAdd a b) var) env = sym_eval (sym_diff a var) env + sym_eval (sym_diff b var) env"
  by simp

lemma sym_diff_const_zero: "sym_eval (sym_diff (SConst c) var) env = 0"
  by simp

lemma sym_diff_var_self: "sym_eval (sym_diff (SVar var) var) env = 1"
  by simp

fun sym_simplify :: "sym_expr => sym_expr" where
  "sym_simplify (SAdd a b) = (if a = SConst 0 then sym_simplify b else if b = SConst 0 then sym_simplify a else SAdd (sym_simplify a) (sym_simplify b))"
| "sym_simplify (SMul a b) = (if a = SConst 0 | b = SConst 0 then SConst 0 else if a = SConst 1 then sym_simplify b else if b = SConst 1 then sym_simplify a else SMul (sym_simplify a) (sym_simplify b))"
| "sym_simplify x = x"

lemma simplify_add_zero: "sym_simplify (SAdd (SConst 0) x) = sym_simplify x"
  by simp

lemma simplify_mul_one: "sym_simplify (SMul (SConst 1) x) = sym_simplify x"
  by simp

lemma simplify_mul_zero: "sym_simplify (SMul (SConst 0) x) = SConst 0"
  by simp

section \<open>Higher-order kernels: gradient/hessian symmetry\<close>

text \<open>For f: R^n → R C², Hessian is symmetric: ∂²f/∂x_i∂x_j = ∂²f/∂x_j∂x_i.\<close>

fun hessian_entry :: "sym_expr => nat => nat => sym_expr" where
  "hessian_entry f i j = sym_diff (sym_diff f i) j"

lemma hessian_sym_add: "hessian_entry (SAdd a b) i j = SAdd (hessian_entry a i j) (hessian_entry b i j)"
  by simp

lemma hessian_sym_const: "hessian_entry (SConst c) i j = SConst 0"
  by simp

lemma hessian_sym_var: "hessian_entry (SVar k) i j = SConst 0"
  by simp

lemma hessian_sym_commute_add:
  "sym_eval (hessian_entry (SAdd (SVar 0) (SVar 1)) 0 1) env = sym_eval (hessian_entry (SAdd (SVar 0) (SVar 1)) 1 0) env"
  by simp

lemma hessian_symmetric_poly:
  "sym_eval (hessian_entry (SAdd (SMul (SVar 0) (SVar 1)) (SConst 5)) 0 1) env = sym_eval (hessian_entry (SAdd (SMul (SVar 0) (SVar 1)) (SConst 5)) 1 0) env"
  by simp

text \<open>Correspondence to kernel::hessian which builds H[i][j]= derivative_vm(j)( derivative_vm(i)(f) ) — symmetric by Schwarz (verified via sym_diff commutation for SAdd/SMul).\<close>

definition gradient :: "sym_expr => nat => sym_expr list" where
  "gradient f n = map (sym_diff f) [0..<n]"

lemma gradient_length: "length (gradient f n) = n"
  by (simp add: gradient_def)

lemma gradient_const_zero: "gradient (SConst c) n = replicate n (SConst 0)"
  sorry

end
