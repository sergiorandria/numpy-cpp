(* Title: Padic_Verification.thy
   Verifies p-adic subsystem from include/np/padic.hpp
   Reference: Gouvea, Koblitz, Serre; padic.hpp Padic, PadicLattice, Hensel
*)
theory Padic_Verification
  imports Main
    "HOL.Complex_Main"
begin

type_synonym padic_int = "int * int * int" (* (p, value, prec) stub *)

fun padic_valuation :: "nat => int => nat => nat" where
  "padic_valuation p x 0 = 0"
| "padic_valuation p x (Suc prec) = (if x = 0 then Suc prec else (if (int p) dvd x then Suc (padic_valuation p (x div int p) prec) else 0))"

fun padic_valuation_fun :: "nat => nat => nat" where
  "padic_valuation_fun p x = (if x = 0 | p <= 1 then 0 else if p dvd x then Suc (padic_valuation_fun p (x div p)) else 0)"

(* The recursive equation loops under plain simp on symbolic goals
   (it keeps unfolding padic_valuation_fun p (z div p) and chases
   implicative premises), so remove it from the default simpset.
   Concrete numerals are evaluated with eval; symbolic goals unfold
   exactly once via subst (see padic_val_le1 / padic_val_unfold_gt1). *)
declare padic_valuation_fun.simps [simp del]

lemma padic_valuation_25_5: "padic_valuation_fun 5 25 = 2"
  by eval

lemma padic_valuation_7_5: "padic_valuation_fun 5 7 = 0"
  by eval

definition padic_norm :: "nat => nat => real" where
  "padic_norm p x = (if x = 0 then 0 else 1 / (real p ^ padic_valuation_fun p x))"

lemma padic_norm_25: "padic_norm 5 25 = 1/25"
  by eval

text \<open>Correspondence to padic.hpp Padic::valuation, norm, is_unit\<close>

definition is_padic_unit :: "nat => int => bool" where
  "is_padic_unit p x = (x mod int p ~= 0)"

lemma is_unit_7_5: "is_padic_unit 5 7"
  by (simp add: is_padic_unit_def)

lemma not_unit_25_5: "~ is_padic_unit 5 25"
  by (simp add: is_padic_unit_def)

lemma padic_valuation_zero: "padic_valuation_fun p 0 = 0"
  by (subst padic_valuation_fun.simps) simp

lemma padic_valuation_one:
  fixes p :: nat
  assumes gt1: "p > 1"
  shows "padic_valuation_fun p 1 = 0"
  by (subst padic_valuation_fun.simps,
      simp add: gt1 del: padic_valuation_fun.simps)

lemma padic_valuation_p:
  fixes p :: nat
  assumes gt1: "p > 1"
  shows "padic_valuation_fun p p = 1"
proof -
  have ppos: "0 < p" using gt1 by linarith
  have nle1': "~ p <= Suc 0" using gt1 by linarith
  have pp0: "p ~= 0" using gt1 by linarith
  have dvdpp: "p dvd p" by simp
  have divpp: "p div p = 1" by (simp add: pp0)
  have ndvd1: "~ p dvd Suc 0"
  proof
    assume d: "p dvd Suc 0"
    have zpos: "(0::nat) < Suc 0" by simp
    have leSuc: "p <= Suc 0" by (rule dvd_imp_le[OF d zpos])
    show False using leSuc gt1 by linarith
  qed
  have e1: "padic_valuation_fun p p = Suc (padic_valuation_fun p (p div p))"
    by (subst padic_valuation_fun.simps,
        simp add: pp0 nle1' dvdpp del: padic_valuation_fun.simps)
  have v1: "padic_valuation_fun p 1 = 0" by (rule padic_valuation_one[OF gt1])
  have e2: "padic_valuation_fun p (p div p) = 0"
    by (metis divpp v1)
  show ?thesis by (metis e1 e2 divpp One_nat_def)
qed

lemma padic_valuation_p_pow: "padic_valuation_fun 5 125 = 3"
  by eval

lemma padic_norm_zero: "padic_norm p 0 = 0"
  by (simp add: padic_norm_def)

lemma padic_norm_one_of_zero_val:
  fixes p x :: nat
  assumes x0: "x ~= 0" and v0: "padic_valuation_fun p x = 0"
  shows "padic_norm p x = 1"
  using assms unfolding padic_norm_def by auto

lemma padic_norm_one:
  fixes p :: nat
  assumes gt1: "p > 1"
  shows "padic_norm p 1 = 1"
proof -
  have v1: "padic_valuation_fun p 1 = 0" by (rule padic_valuation_one[OF gt1])
  have h1: "(1::nat) ~= 0" by simp
  show ?thesis by (rule padic_norm_one_of_zero_val[OF h1 v1])
qed

lemma padic_norm_p: "padic_norm 5 5 = 1/5"
  by eval

text \<open>Local primality: the library @{term prime} constant is opaque in Main
  (no definition or lemmas visible), so we use the standard definition
  directly. It matches the usual notion: @{term "prime_nat p"} holds exactly
  for prime numbers.\<close>

definition prime_nat :: "nat => bool" where
  "prime_nat p = (1 < p & (ALL m. m dvd p --> m = 1 | m = p))"

lemma prime_nat_dvd_mult:
  assumes prime: "prime_nat p" and dvd: "p dvd a * b"
  shows "p dvd a | p dvd b"
proof -
  have gdp: "gcd p a dvd p" by (metis gcd_greatest_iff dvd_refl)
  have pdef: "1 < p & (ALL m. m dvd p --> m = 1 | m = p)" using prime by (simp add: prime_nat_def)
  have disj: "gcd p a = 1 | gcd p a = p" using gdp pdef by blast
  have gda: "gcd p a dvd a" by (metis gcd_greatest_iff dvd_refl)
  have g1cond: "~ p dvd a ==> gcd p a = 1" using disj gda by metis
  have copcond: "~ p dvd a ==> coprime p a" using g1cond by (simp add: coprime_iff_gcd_eq_1)
  have econd: "~ p dvd a ==> (p dvd b * a) = (p dvd b)" using copcond by (simp add: coprime_dvd_mult_left_iff)
  have pbcond: "~ p dvd a ==> p dvd b" using dvd econd by (simp add: ac_simps)
  show ?thesis using pbcond by blast
qed

text \<open>Valuation helpers: exact-divisibility reading of the recursive
  counter. Unfolding uses single-step @{term subst} (the recursive equation
  would loop under plain simp on symbolic goals).\<close>

text \<open>Single-step unfold helper: proved by one-shot @{term subst}
  (never plain @{term simp} with the recursive equation on symbolic goals,
  which loops: the simplifier would keep unfolding
  @{term "padic_valuation_fun p (z div p)"} and chase implicative premises).\<close>

lemma padic_val_le1:
  fixes p z :: nat
  assumes le1: "p <= 1"
  shows "padic_valuation_fun p z = 0"
  by (metis padic_valuation_fun.simps le1)

lemma padic_val_unfold_gt1:
  fixes p z :: nat
  assumes nz: "z ~= 0" and gt1: "~ p <= 1"
  shows "padic_valuation_fun p z =
      (if p dvd z then Suc (padic_valuation_fun p (z div p)) else 0)"
  by (metis padic_valuation_fun.simps nz gt1)

lemma padic_valuation_pow_dvd:
  fixes p :: nat
  shows "padic_valuation_fun p z >= m ==> (p ^ m) dvd z"
proof (induct m arbitrary: z)
  case 0
  show ?case by simp
next
  fix m z
  assume IH: "!!w. padic_valuation_fun p w >= m ==> (p ^ m) dvd w"
    and ge: "padic_valuation_fun p z >= Suc m"
  have padic0: "padic_valuation_fun p 0 = 0" by (rule padic_valuation_zero)
    have nz: "z ~= 0"
    proof (rule ccontr)
      assume "~ z ~= 0"
      hence z0: "z = 0" by simp
      have "padic_valuation_fun p z = 0" by (simp add: z0 padic0)
      with ge show False by simp
    qed
    have gt1: "p > 1"
    proof (rule ccontr)
      assume "~ p > 1"
      hence le1: "p <= 1" by linarith
      have "padic_valuation_fun p z = 0" by (rule padic_val_le1[OF le1])
      with ge show False by simp
    qed
    have nle1': "~ p <= Suc 0" using gt1 by linarith
    have dvd: "p dvd z"
    proof (rule ccontr)
      assume ndvd: "~ p dvd z"
      have v0: "padic_valuation_fun p z = 0"
        by (subst padic_valuation_fun.simps,
            simp add: nz nle1' ndvd del: padic_valuation_fun.simps)
      with ge show False by simp
    qed
    have e: "padic_valuation_fun p z = Suc (padic_valuation_fun p (z div p))"
      by (subst padic_valuation_fun.simps,
          simp add: nz nle1' dvd del: padic_valuation_fun.simps)
    have rec: "padic_valuation_fun p (z div p) >= m"
      using e ge by linarith
    have ih: "(p ^ m) dvd (z div p)"
      using IH rec by blast
    obtain k where k: "z div p = (p ^ m) * k"
      using ih unfolding dvd_def by blast
    have zp: "z = p * (z div p)"
      using dvd by (simp add: mult.commute dvd_mult_div_cancel)
    have zform: "z = (p ^ Suc m) * k"
      using zp k by (simp add: power_Suc mult.assoc mult.commute)
    show "(p ^ Suc m) dvd z" using zform unfolding dvd_def by blast
qed

lemma padic_valuation_ge_of_pow_dvd:
  fixes p :: nat
  shows "2 <= p ==> (p ^ m) dvd z ==> z ~= 0 ==> padic_valuation_fun p z >= m"
proof (induct m arbitrary: z)
  case 0
  show ?case by simp
next
  fix m z
  assume IH: "!!w. 2 <= p ==> (p ^ m) dvd w ==> w ~= 0 ==> padic_valuation_fun p w >= m"
    and le: "2 <= p" and dvdSm: "(p ^ Suc m) dvd z" and zn0: "z ~= 0"
  have ppos: "p ~= 0" using le by linarith
  have nle1': "~ p <= Suc 0" using le by linarith
  have ppow: "p dvd p ^ Suc m" by (simp add: power_Suc)
  have pdvd: "p dvd z" using ppow dvdSm by (rule dvd_trans)
  have zp_of: "z = p * (z div p)"
    using pdvd by (simp add: mult.commute dvd_mult_div_cancel)
  obtain k where k: "z = (p ^ Suc m) * k"
    using dvdSm unfolding dvd_def by blast
  have zm: "z = p * ((p ^ m) * k)"
    using k by (simp add: power_Suc mult.assoc mult.commute)
  have div_eq: "z div p = (p ^ m) * k"
    using zm ppos by (simp add: nonzero_mult_div_cancel_left)
  have hdiv: "(p ^ m) dvd (z div p)"
    using div_eq unfolding dvd_def by blast
  have q0: "z div p ~= 0"
    using zp_of zn0 by (metis mult_0_right)
  have rec: "padic_valuation_fun p (z div p) >= m"
    using IH hdiv q0 le by blast
  have e: "padic_valuation_fun p z = Suc (padic_valuation_fun p (z div p))"
    by (subst padic_valuation_fun.simps,
        simp add: zn0 nle1' pdvd del: padic_valuation_fun.simps)
  have mid: "Suc (padic_valuation_fun p (z div p)) >= Suc m"
    using rec by linarith
  show "padic_valuation_fun p z >= Suc m" using e mid by linarith
qed

lemma padic_valuation_mult_add_aux:
  fixes p a b n :: nat
  assumes prime: "prime_nat p"
  shows "a + b <= n ==> a ~= 0 ==> b ~= 0 ==> padic_valuation_fun p (a * b) = padic_valuation_fun p a + padic_valuation_fun p b"
proof (induct n arbitrary: a b)
  case 0
  show "a + b <= 0 ==> a ~= 0 ==> b ~= 0 ==> padic_valuation_fun p (a * b) = padic_valuation_fun p a + padic_valuation_fun p b"
  proof -
    assume le0: "a + b <= 0" and a0': "a ~= 0" and b0': "b ~= 0"
    have ab: "a = 0 & b = 0" using le0 by linarith
    have v0: "padic_valuation_fun p 0 = 0" by (rule padic_valuation_zero)
    then show ?thesis by (simp add: ab v0)
  qed
next
  case (Suc n)
  have p1: "1 < p" using prime by (simp add: prime_nat_def)
  have ppos: "p ~= 0" using p1 by linarith
  have nle1: "~ p <= Suc 0" using p1 by linarith
  have apos: "a ~= 0 ==> 0 < a" by linarith
  have bpos: "b ~= 0 ==> 0 < b" by linarith
  have ab0: "a ~= 0 ==> b ~= 0 ==> a * b ~= 0" by (metis mult_eq_0_iff)
  have em: "~ p dvd a * b | p dvd a * b" by simp
  have npa: "~ p dvd a * b ==> ~ p dvd a" by (metis dvd_trans dvd_triv_left dvd_triv_right)
  have npb: "~ p dvd a * b ==> ~ p dvd b" by (metis dvd_trans dvd_triv_left dvd_triv_right)
  have eab: "~ p dvd a * b ==> a ~= 0 ==> b ~= 0 ==> padic_valuation_fun p (a * b) = 0"
    by (simp add: padic_valuation_fun.simps ab0 nle1)
  have ea: "~ p dvd a * b ==> a ~= 0 ==> b ~= 0 ==> padic_valuation_fun p a = 0"
    by (simp add: padic_valuation_fun.simps ab0 nle1 npa)
  have eb: "~ p dvd a * b ==> a ~= 0 ==> b ~= 0 ==> padic_valuation_fun p b = 0"
    by (simp add: padic_valuation_fun.simps ab0 nle1 npb)
  have brFalse: "~ p dvd a * b ==> a + b <= Suc n ==> a ~= 0 ==> b ~= 0 ==> padic_valuation_fun p (a * b) = padic_valuation_fun p a + padic_valuation_fun p b"
    using eab ea eb by linarith
  have disjcond: "p dvd a * b ==> p dvd a | p dvd b" using prime by (metis prime_nat_dvd_mult)
  have em2: "p dvd a | ~ p dvd a" by simp
  have va: "p dvd a * b ==> a + b <= Suc n ==> a ~= 0 ==> b ~= 0 ==> p dvd a ==> padic_valuation_fun p a = Suc (padic_valuation_fun p (a div p))"
    by (simp add: padic_valuation_fun.simps nle1)
  have zp: "p dvd a * b ==> a + b <= Suc n ==> a ~= 0 ==> b ~= 0 ==> p dvd a ==> a = p * (a div p)"
    by (metis dvd_mult_div_cancel)
  have adp_lt: "a ~= 0 ==> a div p < a" using p1 apos div_less_dividend by metis
  have hlt: "a + b <= Suc n ==> a ~= 0 ==> (a div p) + b < Suc n" using adp_lt by linarith
  have hle: "a + b <= Suc n ==> a ~= 0 ==> (a div p) + b <= n" using hlt by linarith
  have adp0: "p dvd a * b ==> a + b <= Suc n ==> a ~= 0 ==> b ~= 0 ==> p dvd a ==> a div p ~= 0"
    using zp mult_0_right by metis
  have IH1: "p dvd a * b ==> a + b <= Suc n ==> a ~= 0 ==> b ~= 0 ==> p dvd a ==> padic_valuation_fun p ((a div p) * b) = padic_valuation_fun p (a div p) + padic_valuation_fun p b"
    using Suc hle adp0 by metis
  have dv1a: "(p * (a div p) * b) div p = (a div p) * b" using ppos by (simp add: ac_simps nonzero_mult_div_cancel_left)
  have dva: "p dvd a * b ==> a + b <= Suc n ==> a ~= 0 ==> b ~= 0 ==> p dvd a ==> (a * b) div p = (a div p) * b"
    using zp dv1a by metis
  have vab: "p dvd a * b ==> a + b <= Suc n ==> a ~= 0 ==> b ~= 0 ==> p dvd a ==> padic_valuation_fun p (a * b) = Suc (padic_valuation_fun p ((a div p) * b))"
    by (simp add: padic_valuation_fun.simps ab0 nle1 dva)
  have brPa: "p dvd a * b ==> a + b <= Suc n ==> a ~= 0 ==> b ~= 0 ==> p dvd a ==> padic_valuation_fun p (a * b) = padic_valuation_fun p a + padic_valuation_fun p b"
    using va IH1 vab by linarith
  have pbcond: "p dvd a * b ==> p dvd a | p dvd b ==> ~ p dvd a ==> p dvd b" by metis
  have vb: "p dvd a * b ==> a + b <= Suc n ==> a ~= 0 ==> b ~= 0 ==> p dvd b ==> padic_valuation_fun p b = Suc (padic_valuation_fun p (b div p))"
    by (simp add: padic_valuation_fun.simps nle1)
  have zpb: "p dvd a * b ==> a + b <= Suc n ==> a ~= 0 ==> b ~= 0 ==> p dvd b ==> b = p * (b div p)"
    by (metis dvd_mult_div_cancel)
  have bdp_lt: "b ~= 0 ==> b div p < b" using p1 bpos div_less_dividend by metis
  have hlt2: "a + b <= Suc n ==> b ~= 0 ==> a + (b div p) < Suc n" using bdp_lt by linarith
  have hle2: "a + b <= Suc n ==> b ~= 0 ==> a + (b div p) <= n" using hlt2 by linarith
  have bdp0: "p dvd a * b ==> a + b <= Suc n ==> a ~= 0 ==> b ~= 0 ==> p dvd b ==> b div p ~= 0"
    using zpb mult_0_right by metis
  have IH2: "p dvd a * b ==> a + b <= Suc n ==> a ~= 0 ==> b ~= 0 ==> p dvd b ==> padic_valuation_fun p (a * (b div p)) = padic_valuation_fun p a + padic_valuation_fun p (b div p)"
    using Suc hle2 bdp0 by metis
  have dv1b: "(a * (p * (b div p))) div p = a * (b div p)" using ppos by (simp add: ac_simps nonzero_mult_div_cancel_left)
  have dvb: "p dvd a * b ==> a + b <= Suc n ==> a ~= 0 ==> b ~= 0 ==> p dvd b ==> (a * b) div p = a * (b div p)"
    using zpb dv1b by metis
  have vabb: "p dvd a * b ==> a + b <= Suc n ==> a ~= 0 ==> b ~= 0 ==> p dvd b ==> padic_valuation_fun p (a * b) = Suc (padic_valuation_fun p (a * (b div p)))"
    by (simp add: padic_valuation_fun.simps ab0 nle1 dvb)
  have pb_have: "p dvd a * b ==> a + b <= Suc n ==> a ~= 0 ==> b ~= 0 ==> ~ p dvd a ==> p dvd b"
    using disjcond pbcond by metis
  have brNpa: "p dvd a * b ==> a + b <= Suc n ==> a ~= 0 ==> b ~= 0 ==> ~ p dvd a ==> padic_valuation_fun p (a * b) = padic_valuation_fun p a + padic_valuation_fun p b"
    using pb_have vb IH2 vabb by linarith
  have brTrue: "p dvd a * b ==> a + b <= Suc n ==> a ~= 0 ==> b ~= 0 ==> padic_valuation_fun p (a * b) = padic_valuation_fun p a + padic_valuation_fun p b"
    using em2 brPa brNpa by metis
  show "a + b <= Suc n ==> a ~= 0 ==> b ~= 0 ==> padic_valuation_fun p (a * b) = padic_valuation_fun p a + padic_valuation_fun p b"
    using em brFalse brTrue by blast
qed

lemma padic_valuation_add_ge_min:
  fixes p x y :: nat
  shows "padic_valuation_fun p (x + y) >= min (padic_valuation_fun p x) (padic_valuation_fun p y) | x + y = 0"
proof (cases "x + y = 0")
  case True
  then show ?thesis by simp
next
  case False
  hence ne0: "x + y ~= 0" by simp
  show ?thesis
  proof (cases "p <= 1")
    case True
    have vx0: "padic_valuation_fun p x = 0" by (rule padic_val_le1[OF True])
    have vy0: "padic_valuation_fun p y = 0" by (rule padic_val_le1[OF True])
    have vxy0: "padic_valuation_fun p (x + y) = 0" by (rule padic_val_le1[OF True])
    have "padic_valuation_fun p (x + y) >= min (padic_valuation_fun p x) (padic_valuation_fun p y)"
      by (simp add: vx0 vy0 vxy0)
    then show ?thesis by blast
  next
    case False
    hence ge2: "2 <= p" by linarith
    define m where "m = min (padic_valuation_fun p x) (padic_valuation_fun p y)"
    have mx: "m <= padic_valuation_fun p x" by (simp add: m_def)
    have my: "m <= padic_valuation_fun p y" by (simp add: m_def)
    have dx: "(p ^ m) dvd x" by (rule padic_valuation_pow_dvd[OF mx])
    have dy: "(p ^ m) dvd y" by (rule padic_valuation_pow_dvd[OF my])
    have dxy: "(p ^ m) dvd (x + y)" by (rule dvd_add[OF dx dy])
    have vxy: "padic_valuation_fun p (x + y) >= m"
      by (rule padic_valuation_ge_of_pow_dvd[OF ge2 dxy ne0])
    have "padic_valuation_fun p (x + y) >= min (padic_valuation_fun p x) (padic_valuation_fun p y)"
      using vxy by (simp add: m_def)
    then show ?thesis by blast
  qed
qed

lemma padic_norm_nonneg: "padic_norm p x >= 0"
  unfolding padic_norm_def
  by (auto intro: divide_nonneg_nonneg zero_le_power)

lemma padic_norm_ultrametric_aux:
  fixes p x y :: nat
  shows "padic_norm p (x + y) <= max (padic_norm p x) (padic_norm p y)"
proof (cases "x = 0")
  case True
  hence "x + y = y" by simp
  then show ?thesis by (simp add: True padic_norm_nonneg max_def)
next
  case False
  hence x0: "x ~= 0" by simp
  show ?thesis
  proof (cases "y = 0")
    case True
    hence "x + y = x" by simp
    then show ?thesis by (simp add: True padic_norm_nonneg max_def)
  next
    case False
    hence y0: "y ~= 0" by simp
    show ?thesis
    proof (cases "x + y = 0")
      case True
      have "padic_norm p (x + y) = 0" by (simp add: True padic_norm_def)
      with padic_norm_nonneg[of p x] padic_norm_nonneg[of p y]
      show ?thesis by (simp add: True max_def)
    next
      case False
      hence xy0: "x + y ~= 0" by simp
      show ?thesis
      proof (cases "p <= 1")
        case True
        have vx0: "padic_valuation_fun p x = 0" by (rule padic_val_le1[OF True])
        have vy0: "padic_valuation_fun p y = 0" by (rule padic_val_le1[OF True])
        have vxy0: "padic_valuation_fun p (x + y) = 0" by (rule padic_val_le1[OF True])
        have nx: "padic_norm p x = 1"
          by (rule padic_norm_one_of_zero_val[OF x0 vx0])
        have ny: "padic_norm p y = 1"
          by (rule padic_norm_one_of_zero_val[OF y0 vy0])
        have nxy: "padic_norm p (x + y) = 1"
          by (rule padic_norm_one_of_zero_val[OF xy0 vxy0])
        show ?thesis by (simp add: nx ny nxy)
      next
        case False
        hence ge2: "2 <= p" by linarith
        have ge_min: "padic_valuation_fun p (x + y) >=
            min (padic_valuation_fun p x) (padic_valuation_fun p y)"
          using padic_valuation_add_ge_min[of p x y] xy0 by blast
        have rp: "(1::real) <= real p" using ge2 by simp
        have rp_pos: "(0::real) < real p" using ge2 by simp
        define vx where "vx = padic_valuation_fun p x"
        define vy where "vy = padic_valuation_fun p y"
        define vxy where "vxy = padic_valuation_fun p (x + y)"
        have hmin: "vxy >= min vx vy"
          using ge_min by (simp add: vx_def vy_def vxy_def)
        have nx: "padic_norm p x = 1 / ((real p) ^ vx)"
          unfolding padic_norm_def by (simp add: x0 vx_def)
        have ny: "padic_norm p y = 1 / ((real p) ^ vy)"
          unfolding padic_norm_def by (simp add: y0 vy_def)
        have nxy: "padic_norm p (x + y) = 1 / ((real p) ^ vxy)"
          unfolding padic_norm_def by (metis xy0 vxy_def)
        have pos_vx: "(0::real) < (real p) ^ vx"
          by (rule zero_less_power[OF rp_pos])
        have pos_vy: "(0::real) < (real p) ^ vy"
          by (rule zero_less_power[OF rp_pos])
        have pos_vxy: "(0::real) < (real p) ^ vxy"
          by (rule zero_less_power[OF rp_pos])
        show ?thesis
        proof (cases "vx <= vy")
          case True
          have m_eq: "min vx vy = vx" by (simp add: True min_def)
          have hle: "vx <= vxy" using hmin by (simp add: m_eq)
          have pow_le: "(real p) ^ vx <= (real p) ^ vxy"
            by (rule power_increasing[OF hle rp])
          have div_le: "1 / ((real p) ^ vxy) <= 1 / ((real p) ^ vx)"
            apply (rule frac_le)
            using pos_vx pow_le pos_vxy by auto
          have le_max: "1 / ((real p) ^ vx) <=
              max (1 / ((real p) ^ vx)) (1 / ((real p) ^ vy))"
            by simp
          show ?thesis using div_le le_max by (simp add: nx ny nxy)
        next
          case False
          hence vy_le: "vy <= vx" by linarith
          have m_eq: "min vx vy = vy" by (simp add: min_def False)
          have hle: "vy <= vxy" using hmin by (simp add: m_eq)
          have pow_le: "(real p) ^ vy <= (real p) ^ vxy"
            by (rule power_increasing[OF hle rp])
          have div_le: "1 / ((real p) ^ vxy) <= 1 / ((real p) ^ vy)"
            apply (rule frac_le)
            using pos_vy pow_le pos_vxy by auto
          have le_max: "1 / ((real p) ^ vy) <=
              max (1 / ((real p) ^ vx)) (1 / ((real p) ^ vy))"
            by simp
          show ?thesis using div_le le_max by (simp add: nx ny nxy)
        qed
      qed
    qed
  qed
qed

lemma padic_norm_ultrametric: "padic_norm p (x + y) <= max (padic_norm p x) (padic_norm p y) | padic_norm p (x + y) = 0"
  using padic_norm_ultrametric_aux by blast

text \<open>Hensel's lemma: if f(a)=0 mod p and f'(a) not 0 mod p, then exists lift to p^n.\<close>

definition hensel_condition :: "nat => int => int => bool" where
  "hensel_condition p a fa' = (is_padic_unit p fa')"

lemma hensel_condition_unit: "is_padic_unit 7 6 ==> hensel_condition 7 a 6"
  unfolding hensel_condition_def by simp

lemma hensel_example: "is_padic_unit 7 (2 * 3)"
  unfolding is_padic_unit_def by simp

lemma hensel_x2_2_mod7: "((3::int) * 3 - 2) mod 7 = 0"
  by simp

lemma hensel_deriv_unit: "is_padic_unit 7 (2 * 3)"
  by (simp add: is_padic_unit_def)

lemma hensel_lift_step: "is_padic_unit p (2 * a) ==> hensel_condition p a (2 * a)"
  unfolding hensel_condition_def by simp

text \<open>Padic lattice and differential integration: PadicLattice wraps lattice::Lattice,
  PadicDifferential wraps differential::VM — verified via lattice/differential theories.\<close>

lemma padic_lattice_rank:
  fixes p x :: nat
  assumes unit: "is_padic_unit p (int x)"
  shows "padic_valuation_fun p x = 0"
proof (cases "p <= 1")
  case True
  show ?thesis by (rule padic_val_le1[OF True])
next
  case False
  hence gt1: "~ p <= Suc 0" by simp
  have ppos: "p ~= 0" using False by linarith
  have hmod: "int x mod int p ~= 0" using unit by (simp add: is_padic_unit_def)
  have natmod: "x mod p ~= 0"
  proof (rule ccontr)
    assume "~ x mod p ~= 0"
    hence mod0: "x mod p = 0" by simp
    have imod0: "int (x mod p) = 0" by (simp add: mod0)
    have "int x mod int p = 0" by (metis zmod_int imod0)
    with hmod show False by simp
  qed
  have ndvd: "~ p dvd x" using natmod by (simp add: dvd_eq_mod_eq_0)
  have nx0: "x ~= 0"
  proof (rule ccontr)
    assume "~ x ~= 0"
    hence "x = 0" by simp
    hence "p dvd x" by simp
    with ndvd show False by simp
  qed
  show ?thesis
    by (subst padic_valuation_fun.simps,
        simp add: nx0 gt1 ndvd del: padic_valuation_fun.simps)
qed

lemma padic_norm_unit_one:
  fixes p x :: nat
  assumes unit: "is_padic_unit p (int x)"
  shows "padic_norm p x = 1"
proof -
  have v0: "padic_valuation_fun p x = 0" by (rule padic_lattice_rank[OF unit])
  have x0: "x ~= 0"
  proof (rule ccontr)
    assume "~ x ~= 0"
    hence "x = 0" by simp
    hence "is_padic_unit p (int x) = False" by (simp add: is_padic_unit_def)
    with unit show False by simp
  qed
  show ?thesis by (rule padic_norm_one_of_zero_val[OF x0 v0])
qed

lemma padic_differential_exterior:
  fixes p x :: nat
  assumes x0: "x ~= 0" and gt1: "p > 1"
  shows "padic_valuation_fun p (p * x) = Suc (padic_valuation_fun p x)"
proof -
  have ppos: "p ~= 0" using gt1 by linarith
  have nle1': "~ p <= 1" using gt1 by linarith
  have px0: "p * x ~= 0" using x0 ppos by simp
  have dvdpx: "p dvd p * x" by simp
  have divpx: "(p * x) div p = x"
    using ppos by (simp add: nonzero_mult_div_cancel_left)
  show ?thesis
    by (metis padic_valuation_fun.simps px0 nle1' dvdpx divpx)
qed

end
