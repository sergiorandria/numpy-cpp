(* Title: Window_Verification.thy
   Verifies window functions from include/np/window.hpp
   (bartlett, blackman, hamming, hanning/hann, kaiser).
*)
theory Window_Verification
  imports Main
    "HOL.Complex_Main"
begin

section \<open>Bartlett (triangular) window\<close>

definition bartlett_coeff :: "real => real => real" where
  "bartlett_coeff M n = 2 / (M - 1) * ((M - 1) / 2 - abs (n - (M - 1) / 2))"

definition bartlett_window :: "nat => real list" where
  "bartlett_window M =
    (if M = 0 then [] else if M = 1 then [1]
     else map (bartlett_coeff (real M) o real) [0..<M])"

lemma bartlett_length: "length (bartlett_window M) = M"
  by (simp add: bartlett_window_def)

lemma bartlett_zero: "bartlett_window 0 = []"
  by (simp add: bartlett_window_def)

lemma bartlett_one: "bartlett_window 1 = [1]"
  by (simp add: bartlett_window_def)

lemma bartlett_coeff_sym:
  "bartlett_coeff M (M - 1 - n) = bartlett_coeff M n"
proof -
  have h: "(M - 1 - n) - (M - 1) / 2 = - (n - (M - 1) / 2)"
    by (simp add: algebra_simps field_simps)
  have a: "abs ((M - 1 - n) - (M - 1) / 2) = abs (n - (M - 1) / 2)"
  proof -
    show ?thesis
      apply (subst h)
      apply (simp add: abs_minus)
      done
  qed
  show ?thesis unfolding bartlett_coeff_def using a by simp
qed

lemma bartlett_sym:
  assumes "M > 1" and "n < M"
  shows "(bartlett_window M) ! n = (bartlett_window M) ! (M - 1 - n)"
proof -
  have M1: "M ~= 0" "M ~= 1" using assms by auto
  have nth: "!!k. k < M ==> (bartlett_window M) ! k = bartlett_coeff (real M) (real k)"
    using M1 by (simp add: bartlett_window_def o_def nth_map_upt)
  have n_le: "n <= M - 1" using assms by linarith
  have one_le: "Suc 0 <= M" using assms by linarith
  have r1: "real (M - 1 - n) = real (M - 1) - real n"
    by (rule of_nat_diff[OF n_le])
  have r2: "real (M - 1) = real M - 1"
    by (simp add: of_nat_diff[OF one_le])
  have arg: "real (M - 1 - n) = real M - 1 - real n"
    using r1 r2 by linarith
  show ?thesis
    using assms nth arg bartlett_coeff_sym by simp
qed

lemma bartlett_5_endpoints:
  "(bartlett_window 5) ! 0 = 0 & (bartlett_window 5) ! 2 = 1 & (bartlett_window 5) ! 4 = 0"
  unfolding bartlett_window_def bartlett_coeff_def o_def by simp

section \<open>Blackman window\<close>

definition blackman_coeff :: "real => real => real" where
  "blackman_coeff M n = 0.42 - 0.5 * cos (2 * pi * n / (M - 1)) + 0.08 * cos (2 * (2 * pi * n / (M - 1)))"

definition blackman_window :: "nat => real list" where
  "blackman_window M =
    (if M = 0 then [] else if M = 1 then [1]
     else map (blackman_coeff (real M) o real) [0..<M])"

lemma blackman_length: "length (blackman_window M) = M"
  by (simp add: blackman_window_def)

lemma blackman_zero: "blackman_window 0 = []"
  by (simp add: blackman_window_def)

lemma blackman_one: "blackman_window 1 = [1]"
  by (simp add: blackman_window_def)

lemma cos_2pi_sub: "cos (2 * pi - t) = cos t"
  by simp

lemma cos_4pi_sub: "cos (4 * pi - y) = cos y"
proof -
  have e: "(4::real) * pi - y = (2 * pi - y) + 2 * pi"
    by (simp add: algebra_simps)
  show ?thesis
    apply (subst e)
    apply (subst cos_periodic)
    apply (subst cos_2pi_minus)
    apply (rule refl)
    done
qed

lemma blackman_coeff_sym:
  assumes "M > 1"
  shows "blackman_coeff M (M - 1 - n) = blackman_coeff M n"
proof -
  have arg: "2 * pi * (M - 1 - n) / (M - 1) = 2 * pi - 2 * pi * n / (M - 1)"
    using assms unfolding blackman_coeff_def by (simp add: field_simps algebra_simps)
  have arg2: "2 * (2 * pi * (M - 1 - n) / (M - 1)) = 4 * pi - 2 * (2 * pi * n / (M - 1))"
    using assms by (simp add: field_simps algebra_simps)
  show ?thesis
    unfolding blackman_coeff_def arg arg2 by (simp add: cos_2pi_sub cos_4pi_sub)
qed

lemma blackman_sym:
  assumes "M > 1" and "n < M"
  shows "(blackman_window M) ! n = (blackman_window M) ! (M - 1 - n)"
proof -
  have M1: "M ~= 0" "M ~= 1" using assms by auto
  have nth: "!!k. k < M ==> (blackman_window M) ! k = blackman_coeff (real M) (real k)"
    using M1 by (simp add: blackman_window_def o_def nth_map_upt)
  have n_le: "n <= M - 1" using assms by linarith
  have one_le: "Suc 0 <= M" using assms by linarith
  have r1: "real (M - 1 - n) = real (M - 1) - real n"
    by (rule of_nat_diff[OF n_le])
  have r2: "real (M - 1) = real M - 1"
    by (simp add: of_nat_diff[OF one_le])
  have arg: "real (M - 1 - n) = real M - 1 - real n"
    using r1 r2 by linarith
  have lt: "M - 1 - n < M" using assms by linarith
  show ?thesis
    using assms nth arg lt blackman_coeff_sym by simp
qed

lemma blackman_5_peak: "(blackman_window 5) ! 2 > 0.8"
  unfolding blackman_window_def blackman_coeff_def o_def by simp

section \<open>Hamming window\<close>

definition hamming_coeff :: "real => real => real" where
  "hamming_coeff M n = 0.54 - 0.46 * cos (2 * pi * n / (M - 1))"

definition hamming_window :: "nat => real list" where
  "hamming_window M =
    (if M = 0 then [] else if M = 1 then [1]
     else map (hamming_coeff (real M) o real) [0..<M])"

lemma hamming_length: "length (hamming_window M) = M"
  by (simp add: hamming_window_def)

lemma hamming_zero: "hamming_window 0 = []"
  by (simp add: hamming_window_def)

lemma hamming_one: "hamming_window 1 = [1]"
  by (simp add: hamming_window_def)

lemma hamming_coeff_sym:
  assumes "M > 1"
  shows "hamming_coeff M (M - 1 - n) = hamming_coeff M n"
proof -
  have arg: "2 * pi * (M - 1 - n) / (M - 1) = 2 * pi - 2 * pi * n / (M - 1)"
    using assms unfolding hamming_coeff_def by (simp add: field_simps algebra_simps)
  show ?thesis
    unfolding hamming_coeff_def arg by (simp add: cos_2pi_sub)
qed

lemma hamming_sym:
  assumes "M > 1" and "n < M"
  shows "(hamming_window M) ! n = (hamming_window M) ! (M - 1 - n)"
proof -
  have M1: "M ~= 0" "M ~= 1" using assms by auto
  have nth: "!!k. k < M ==> (hamming_window M) ! k = hamming_coeff (real M) (real k)"
    using M1 by (simp add: hamming_window_def o_def nth_map_upt)
  have n_le: "n <= M - 1" using assms by linarith
  have one_le: "Suc 0 <= M" using assms by linarith
  have r1: "real (M - 1 - n) = real (M - 1) - real n"
    by (rule of_nat_diff[OF n_le])
  have r2: "real (M - 1) = real M - 1"
    by (simp add: of_nat_diff[OF one_le])
  have arg: "real (M - 1 - n) = real M - 1 - real n"
    using r1 r2 by linarith
  have lt: "M - 1 - n < M" using assms by linarith
  show ?thesis
    using assms nth arg lt hamming_coeff_sym by simp
qed

lemma hamming_5_endpoints:
  "(hamming_window 5) ! 0 = 0.08 & (hamming_window 5) ! 2 = 1"
  unfolding hamming_window_def hamming_coeff_def o_def by simp

section \<open>Hanning (Hann) window\<close>

definition hanning_coeff :: "real => real => real" where
  "hanning_coeff M n = 0.5 - 0.5 * cos (2 * pi * n / (M - 1))"

definition hanning_window :: "nat => real list" where
  "hanning_window M =
    (if M = 0 then [] else if M = 1 then [1]
     else map (hanning_coeff (real M) o real) [0..<M])"

definition hann_window :: "nat => real list" where
  "hann_window M = hanning_window M"

lemma hanning_length: "length (hanning_window M) = M"
  by (simp add: hanning_window_def)

lemma hanning_zero: "hanning_window 0 = []"
  by (simp add: hanning_window_def)

lemma hanning_one: "hanning_window 1 = [1]"
  by (simp add: hanning_window_def)

lemma hann_alias: "hann_window M = hanning_window M"
  by (simp add: hann_window_def)

lemma hanning_coeff_sym:
  assumes "M > 1"
  shows "hanning_coeff M (M - 1 - n) = hanning_coeff M n"
proof -
  have arg: "2 * pi * (M - 1 - n) / (M - 1) = 2 * pi - 2 * pi * n / (M - 1)"
    using assms unfolding hanning_coeff_def by (simp add: field_simps algebra_simps)
  show ?thesis
    unfolding hanning_coeff_def arg by (simp add: cos_2pi_sub)
qed

lemma hanning_sym:
  assumes "M > 1" and "n < M"
  shows "(hanning_window M) ! n = (hanning_window M) ! (M - 1 - n)"
proof -
  have M1: "M ~= 0" "M ~= 1" using assms by auto
  have nth: "!!k. k < M ==> (hanning_window M) ! k = hanning_coeff (real M) (real k)"
    using M1 by (simp add: hanning_window_def o_def nth_map_upt)
  have n_le: "n <= M - 1" using assms by linarith
  have one_le: "Suc 0 <= M" using assms by linarith
  have r1: "real (M - 1 - n) = real (M - 1) - real n"
    by (rule of_nat_diff[OF n_le])
  have r2: "real (M - 1) = real M - 1"
    by (simp add: of_nat_diff[OF one_le])
  have arg: "real (M - 1 - n) = real M - 1 - real n"
    using r1 r2 by linarith
  have lt: "M - 1 - n < M" using assms by linarith
  show ?thesis
    using assms nth arg lt hanning_coeff_sym by simp
qed

lemma hanning_8_endpoints:
  "(hanning_window 8) ! 0 = 0 & (hanning_window 8) ! 7 = 0"
  unfolding hanning_window_def hanning_coeff_def o_def by simp

section \<open>Kaiser window (finite-sum I0 model of std::cyl_bessel_i)\<close>

text \<open>The C++ implementation uses std::cyl_bessel_i(0) for I0.
HOL has no Bessel library, so we model I0 by its power series truncated to
10 terms: all summands are nonnegative, hence bessel_I0 is positive,
and the Kaiser shape (length, symmetry, peak 1, beta=0 rectangular) is
verified independently of the truncation order.\<close>

definition bessel_I0 :: "real => real" where
  "bessel_I0 x = sum (%k. ((x * x / 4) ^ k) / (real (fact k) * real (fact k))) {..<10}"

lemma bessel_I0_pos: "bessel_I0 x > 0"
proof -
  define f where "f = (%k::nat. ((x * x / 4) ^ k) / (real (fact k) * real (fact k)))"
  have f0: "f 0 = 1" by (simp add: f_def)
  have nn: "!!k. k : {..<10} - {0} ==> 0 <= f k"
    by (simp add: f_def divide_nonneg_nonneg zero_le_power)
  have fin: "finite ({..<10} :: nat set)" by simp
  have mem: "(0::nat) : {..<10}" by simp
  have le: "f 0 <= sum f {..<10}"
    by (rule member_le_sum[OF mem nn fin])
  have e: "bessel_I0 x = sum f {..<10}"
    by (simp add: bessel_I0_def f_def)
  show ?thesis using e f0 le by linarith
qed

lemma bessel_I0_zero: "bessel_I0 0 = 1"
proof -
  define f where "f = (%k::nat. ((((0::real)) * 0 / 4) ^ k) / (real (fact k) * real (fact k)))"
  have f0: "f 0 = 1" by (simp add: f_def)
  have rest: "sum f {1..<10} = 0"
  proof -
    have z: "!!k. k : {1..<10} ==> f k = 0"
    proof -
      fix k :: nat
      assume kk: "k : {1..<10}"
      obtain m where km: "k = Suc m" using kk by (cases k) auto
      show "f k = 0" unfolding f_def km by simp
    qed
    have e: "sum f {1..<10} = sum (%k::nat. 0) {1..<10}"
      by (rule sum.cong[OF refl z])
    show ?thesis using e by simp
  qed
  have fin: "finite ({..<10} :: nat set)" by simp
  have mem: "(0::nat) : {..<10}" by simp
  have split: "sum f {..<10} = f 0 + sum f (({..<10} :: nat set) - {0})"
    by (rule sum.remove[OF fin mem])
  have eq: "(({..<10} :: nat set) - {0}) = {1..<10}" by auto
  have e: "bessel_I0 0 = sum f {..<10}"
    by (simp add: bessel_I0_def f_def)
  show ?thesis using e f0 rest split eq by simp
qed

definition kaiser_coeff :: "real => real => real => real" where
  "kaiser_coeff M beta n =
    bessel_I0 (beta * sqrt (1 - ((2 * n / (M - 1) - 1) ^ 2))) / bessel_I0 beta"

definition kaiser_window :: "nat => real => real list" where
  "kaiser_window M beta =
    (if M = 0 then [] else if M = 1 then [1]
     else map ((kaiser_coeff (real M) beta) o real) [0..<M])"

lemma kaiser_length: "length (kaiser_window M beta) = M"
  by (simp add: kaiser_window_def)

lemma kaiser_zero: "kaiser_window 0 beta = []"
  by (simp add: kaiser_window_def)

lemma kaiser_one: "kaiser_window 1 beta = [1]"
  by (simp add: kaiser_window_def)

lemma kaiser_coeff_sym:
  assumes "M > 1"
  shows "kaiser_coeff M beta (M - 1 - n) = kaiser_coeff M beta n"
  unfolding kaiser_coeff_def
  apply (rule arg_cong[where f="(%t. bessel_I0 (beta * sqrt t) / bessel_I0 beta)"])
  using assms by (simp add: field_simps algebra_simps power2_eq_square)

lemma kaiser_sym:
  assumes "M > 1" and "n < M"
  shows "(kaiser_window M beta) ! n = (kaiser_window M beta) ! (M - 1 - n)"
proof -
  have M1: "M ~= 0" "M ~= 1" using assms by auto
  have nth: "!!k. k < M ==> (kaiser_window M beta) ! k = kaiser_coeff (real M) beta (real k)"
    using M1 by (simp add: kaiser_window_def o_def nth_map_upt)
  have n_le: "n <= M - 1" using assms by linarith
  have one_le: "Suc 0 <= M" using assms by linarith
  have r1: "real (M - 1 - n) = real (M - 1) - real n"
    by (rule of_nat_diff[OF n_le])
  have r2: "real (M - 1) = real M - 1"
    by (simp add: of_nat_diff[OF one_le])
  have arg: "real (M - 1 - n) = real M - 1 - real n"
    using r1 r2 by linarith
  have lt: "M - 1 - n < M" using assms by linarith
  show ?thesis
    using assms nth arg lt kaiser_coeff_sym by simp
qed

lemma kaiser_center_one:
  assumes "M > 1"
  shows "kaiser_coeff M beta ((M - 1) / 2) = 1"
  unfolding kaiser_coeff_def
  using bessel_I0_pos[of beta] assms
  by (simp add: field_simps algebra_simps divide_self)

lemma kaiser_beta_zero_rect:
  "kaiser_coeff M 0 n = 1"
  unfolding kaiser_coeff_def by (simp add: bessel_I0_zero)

text \<open>Correspondence to include/np/window.hpp: bartlett_window,
blackman_window, hamming_window, hanning_window /
hann_window and kaiser_window mirror the C++ edge cases
(M=0 empty, M=1 singleton, negative M rejected by exception in C++),
the window formulas, and the verified properties (length, symmetry,
endpoints, Kaiser peak/rectangular limit) tested in
tests/test_window.cpp.\<close>

end
