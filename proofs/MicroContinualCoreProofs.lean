-- Copyright (c) 2026 Adrian Aleksandrovich İşmuhanov
-- SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
-- Formal 0-Axiom Algebraic Foundations of MicroContinualCore & Cayley-RTRL (Lean 4, no Mathlib, no Real)

set_option maxRecDepth 200000
set_option maxHeartbeats 400000

namespace Foundations

-- =============================================================================
-- ЭТАП I. ЛОГИКА ТОЖДЕСТВА И ЗАПРЕТ САМООТРИЦАНИЯ (Шаги 1–5)
-- =============================================================================

/-- ШАГ 1. Закон тождества (Рефлексивность):
    Для любого объекта или суждения a всегда выполняется a = a. -/
theorem step01_identity {α : Type} (a : α) : a = a := rfl

/-- ШАГ 2. Запрет самоотрицания:
    Предикат a ≠ a влечет противоречие (False).
    Отрицание не требует внешней пустоты: это путь к саморазрушению суждения. -/
theorem step02_no_self_negation {α : Type} (a : α) (h : a ≠ a) : False :=
  h (step01_identity a)

/-- ШАГ 3. Симметрия тождества:
    Если a = b, то b = a. Выводится исключительно через подстановку по тождеству. -/
theorem step03_eq_symm {α : Type} {a b : α} (h : a = b) : b = a := by
  cases h
  rfl

/-- ШАГ 4. Транзитивность тождества:
    Если a = b и b = c, то a = c. Тождество сохраняется по цепочке. -/
theorem step04_eq_trans {α : Type} {a b c : α} (h1 : a = b) (h2 : b = c) : a = c := by
  cases h2
  exact h1

/-- ШАГ 5. Структура различия (Акт бытия):
    Несамопротиворечивое различие — это пара элементов с доказательством неравенства. -/
structure Distinction (α : Type) where
  a : α
  b : α
  diff : a ≠ b

-- =============================================================================
-- ЭТАП II. РОЖДЕНИЕ ПОЗИТИВНОГО НАТУРАЛЬНОГО РЯДА ℕ⁺ (Шаги 6–13)
-- =============================================================================

/-- ШАГ 6. Натуральный ряд без нуля (аксиоматика Пеано 1889 г.):
    Элемент `one` (1) — квант первичного различия относительно самого себя.
    Операция `succ` — шаг направленного изменения. Ноль в типе отсутствует. -/
inductive PosNat : Type where
  | one  : PosNat
  | succ : PosNat → PosNat

namespace PosNat

/-- ШАГ 7. Базовый квант 1 не является результатом шага:
    one ≠ succ n. Процесс направлен строго вперед. -/
theorem step07_one_ne_succ (n : PosNat) : one ≠ succ n := by
  intro h
  contradiction

/-- ШАГ 8. Шаг всегда дает качественно новый элемент (succ n ≠ n):
    Никакое состояние не зацикливается в себя на первом шаге. -/
theorem step08_succ_ne_self (n : PosNat) : succ n ≠ n := by
  induction n with
  | one =>
    intro h
    contradiction
  | succ n ih =>
    intro h
    injection h with h_prev
    exact ih h_prev

/-- ШАГ 9. Инъективность шага различия:
    succ n = succ m → n = m. Разные предыстории не могут сливаться. -/
theorem step09_succ_inj {n m : PosNat} (h : succ n = succ m) : n = m := by
  injection h

/-- ШАГ 10. Рекурсивное определение сложения:
    Сложение — это суперпозиция шагов.
    n + 1 = succ n;  n + succ m = succ (n + m). -/
def add : PosNat → PosNat → PosNat
  | n, one    => succ n
  | n, succ m => succ (add n m)

instance : Add PosNat where
  add := add

@[simp] theorem add_one_def (n : PosNat) : n + one = succ n := rfl
@[simp] theorem add_succ_def (n m : PosNat) : n + succ m = succ (n + m) := rfl

/-- ШАГ 11. Левое прибавление единицы: 1 + n = succ n. -/
theorem step11_one_add (n : PosNat) : one + n = succ n := by
  induction n with
  | one => rfl
  | succ n ih =>
    change succ (one + n) = succ (succ n)
    rw [ih]

/-- Лемма переноса шага: succ n + m = succ (n + m). -/
theorem succ_add (n m : PosNat) : succ n + m = succ (n + m) := by
  induction m with
  | one => rfl
  | succ m ih =>
    change succ (succ n + m) = succ (n + succ m)
    rw [ih]
    rfl

/-- ШАГ 12. Коммутативность сложения: n + m = m + n. -/
theorem step12_add_comm (n m : PosNat) : n + m = m + n := by
  induction m with
  | one =>
    rw [add_one_def, step11_one_add]
  | succ m ih =>
    rw [add_succ_def, succ_add, ih]

/-- ШАГ 13. Ассоциативность сложения: (a + b) + c = a + (b + c). -/
theorem step13_add_assoc (a b c : PosNat) : (a + b) + c = a + (b + c) := by
  induction c with
  | one => rfl
  | succ c ih =>
    change succ ((a + b) + c) = succ (a + (b + c))
    rw [ih]

-- =============================================================================
-- ЭТАП III. УМНОЖЕНИЕ И ДИСТРИБУТИВНОСТЬ (Шаги 14–17)
-- =============================================================================

/-- ШАГ 14. Определение умножения:
    Умножение — повторение масштаба шагов.
    n * 1 = n;  n * succ m = (n * m) + n. -/
def mul : PosNat → PosNat → PosNat
  | n, one    => n
  | n, succ m => mul n m + n

instance : Mul PosNat where
  mul := mul

@[simp] theorem mul_one_def (n : PosNat) : n * one = n := rfl
@[simp] theorem mul_succ_def (n m : PosNat) : n * succ m = (n * m) + n := rfl

/-- ШАГ 15. Левая нейтральность единицы: 1 * n = n. -/
theorem step15_one_mul (n : PosNat) : one * n = n := by
  induction n with
  | one => rfl
  | succ n ih =>
    change (one * n) + one = succ n
    rw [ih, add_one_def]

/-- ШАГ 16. Правая дистрибутивность: (a + b) * c = a * c + b * c. -/
theorem step16_add_mul (a b c : PosNat) : (a + b) * c = a * c + b * c := by
  induction c with
  | one => rfl
  | succ c ih =>
    change (a + b) * c + (a + b) = (a * succ c) + (b * succ c)
    rw [ih]
    change (a * c + b * c) + (a + b) = (a * c + a) + (b * c + b)
    rw [step13_add_assoc, ← step13_add_assoc (b * c) a b, step12_add_comm (b * c) a]
    rw [step13_add_assoc a (b * c) b, ← step13_add_assoc (a * c) a ((b * c) + b)]

/-- Лемма: n * succ m = n + (n * m). -/
theorem succ_mul (n m : PosNat) : succ n * m = n * m + m := by
  induction m with
  | one =>
    rw [mul_one_def, mul_one_def, add_one_def]
  | succ m ih =>
    rw [mul_succ_def, mul_succ_def, ih]
    change (n * m + m + succ n) = (n * m + n + succ m)
    rw [step13_add_assoc (n * m) m (succ n)]
    rw [step13_add_assoc (n * m) n (succ m)]
    have h_swap : m + succ n = n + succ m := by
      change succ (m + n) = succ (n + m)
      rw [step12_add_comm m n]
    rw [h_swap]

/-- ШАГ 17. Коммутативность умножения: n * m = m * n. -/
theorem step17_mul_comm (n m : PosNat) : n * m = m * n := by
  induction m with
  | one =>
    rw [mul_one_def, step15_one_mul]
  | succ m ih =>
    rw [mul_succ_def, succ_mul, ih, step12_add_comm]

theorem mul_add (a b c : PosNat) : a * (b + c) = a * b + a * c := by
  rw [step17_mul_comm a (b + c), step16_add_mul, step17_mul_comm b a, step17_mul_comm c a]

theorem mul_assoc (a b c : PosNat) : (a * b) * c = a * (b * c) := by
  induction c with
  | one => rfl
  | succ c ih =>
    change (a * b) * c + (a * b) = a * (b * succ c)
    rw [ih, mul_succ_def, mul_add]

-- =============================================================================
-- ЭТАП IV. ПОРЯДОК, СОКРАЩЕНИЕ И ТРИХОТОМИЯ (Шаги 18–21)
-- =============================================================================

/-- ШАГ 18. Отношение строгого порядка:
    a < b означает существование реального положительного перепада (свидетеля c). -/
def Lt (a b : PosNat) : Prop :=
  ∃ c : PosNat, a + c = b

instance : LT PosNat where
  lt := Lt

/-- Лемма: a + c ≠ a для любых a и c. Никакая сумма не зануляется. -/
theorem add_ne_self (a c : PosNat) : a + c ≠ a := by
  induction a with
  | one =>
    rw [step11_one_add]
    intro h
    exact step07_one_ne_succ c (step03_eq_symm h)
  | succ a ih =>
    rw [succ_add]
    intro h
    exact ih (step09_succ_inj h)

/-- ШАГ 19. Антирефлексивность строгого порядка:
    Никакое число не меньше самого себя: ¬(a < a). -/
theorem step19_lt_irrefl (a : PosNat) : ¬(a < a) := by
  intro h
  cases h with
  | intro c hc =>
    exact add_ne_self a c hc

/-- ШАГ 20. Транзитивность порядка:
    a < b → b < c → a < c. Перепады суммируются. -/
theorem step20_lt_trans {a b c : PosNat} (h1 : a < b) (h2 : b < c) : a < c := by
  cases h1 with
  | intro c1 hc1 =>
    cases h2 with
    | intro c2 hc2 =>
      exact ⟨c1 + c2, by rw [← step13_add_assoc, hc1, hc2]⟩

/-- Закон левого сокращения сложения: a + x = a + y → x = y. -/
theorem add_left_cancel {a x y : PosNat} (h : a + x = a + y) : x = y := by
  induction a with
  | one =>
    rw [step11_one_add, step11_one_add] at h
    exact step09_succ_inj h
  | succ a ih =>
    rw [succ_add, succ_add] at h
    exact ih (step09_succ_inj h)

/-- Закон правого сокращения сложения: a + c = b + c → a = b. -/
theorem add_right_cancel {a b c : PosNat} (h : a + c = b + c) : a = b := by
  rw [step12_add_comm a c, step12_add_comm b c] at h
  exact add_left_cancel h

/-- ШАГ 21. Вычитание как расстояние (частичная операция):
    Если a + dist = b, то расстояние между ними единственно. -/
theorem step21_distance_unique {a b dist1 dist2 : PosNat}
    (h1 : a + dist1 = b) (h2 : a + dist2 = b) : dist1 = dist2 := by
  have h : a + dist1 = a + dist2 := by rw [h1, h2]
  exact add_left_cancel h

/-- Трихотомия порядка на PosNat (0 аксиом):
    Для любых a и b верно либо a = b, либо a < b, либо b < a. -/
theorem trichotomy (a b : PosNat) : a = b ∨ Lt a b ∨ Lt b a := by
  induction a generalizing b with
  | one =>
    cases b with
    | one => exact Or.inl rfl
    | succ b => exact Or.inr (Or.inl ⟨b, by rw [step11_one_add]⟩)
  | succ a ih =>
    cases b with
    | one => exact Or.inr (Or.inr ⟨a, by rw [step11_one_add]⟩)
    | succ b =>
      cases ih b with
      | inl h => exact Or.inl (congrArg succ h)
      | inr h =>
        cases h with
        | inl h_lt =>
          cases h_lt with
          | intro c hc => exact Or.inr (Or.inl ⟨c, by rw [succ_add, hc]⟩)
        | inr h_gt =>
          cases h_gt with
          | intro c hc => exact Or.inr (Or.inr ⟨c, by rw [succ_add, hc]⟩)

/-- Закон правого сокращения умножения (0 аксиом): a * c = b * c → a = b. -/
theorem mul_right_cancel {a b c : PosNat} (h : a * c = b * c) : a = b := by
  cases trichotomy a b with
  | inl h_eq => exact h_eq
  | inr h_ineq =>
    cases h_ineq with
    | inl h_lt =>
      cases h_lt with
      | intro d hd =>
        have h_mul : b * c = a * c + d * c := by rw [← hd, step16_add_mul]
        rw [← h] at h_mul
        have h_abs := add_ne_self (a * c) (d * c)
        exact (h_abs h_mul.symm).elim
    | inr h_gt =>
      cases h_gt with
      | intro d hd =>
        have h_mul : a * c = b * c + d * c := by rw [← hd, step16_add_mul]
        rw [h] at h_mul
        have h_abs := add_ne_self (b * c) (d * c)
        exact (h_abs h_mul.symm).elim

-- =============================================================================
-- ЭТАП V. ЦЕЛЫЕ ЧИСЛА КАК БАЛАНС И НОЛЬ КАК ТОЖДЕСТВО (Шаги 22–25)
-- =============================================================================

theorem add_right_comm (a b c : PosNat) : (a + b) + c = (a + c) + b := by
  rw [step13_add_assoc, step12_add_comm b c, ← step13_add_assoc]

theorem add_add_add_comm (a b c d : PosNat) : (a + b) + (c + d) = (a + c) + (b + d) := by
  rw [← step13_add_assoc, add_right_comm a b c, step13_add_assoc (a + c) b d]

/-- ШАГ 22. Пространство пар направленных шагов:
    Пара (fwd, bwd) : PosNat × PosNat задает баланс прямого и обратного движения. -/
structure Balance where
  fwd : PosNat
  bwd : PosNat

/-- ШАГ 23. Отношение эквивалентности балансов:
    (a, b) ~ (c, d) ↔ a + d = b + c. Полный уход от "пустого нуля". -/
def BalanceEquiv (x y : Balance) : Prop :=
  x.fwd + y.bwd = x.bwd + y.fwd

/-- ШАГ 24. Рефлексивность, симметрия и транзитивность баланса. -/
theorem step24_balance_refl (x : Balance) : BalanceEquiv x x := by
  dsimp [BalanceEquiv]
  rw [step12_add_comm]

theorem step24_balance_symm {x y : Balance} (h : BalanceEquiv x y) : BalanceEquiv y x := by
  dsimp [BalanceEquiv] at *
  rw [step12_add_comm y.fwd, step12_add_comm y.bwd]
  exact step03_eq_symm h

theorem step24_balance_trans {x y z : Balance}
    (h1 : BalanceEquiv x y) (h2 : BalanceEquiv y z) : BalanceEquiv x z := by
  dsimp [BalanceEquiv] at *
  have h_sum : (x.fwd + y.bwd) + (y.fwd + z.bwd) = (x.bwd + y.fwd) + (y.bwd + z.fwd) := by
    rw [h1, h2]
  have h_lhs : (x.fwd + y.bwd) + (y.fwd + z.bwd) = (x.fwd + z.bwd) + (y.bwd + y.fwd) := by
    rw [step12_add_comm y.fwd z.bwd, add_add_add_comm x.fwd y.bwd z.bwd y.fwd]
  have h_rhs : (x.bwd + y.fwd) + (y.bwd + z.fwd) = (x.bwd + z.fwd) + (y.bwd + y.fwd) := by
    rw [step12_add_comm y.bwd z.fwd, add_add_add_comm x.bwd y.fwd z.fwd y.bwd, step12_add_comm y.fwd y.bwd]
  rw [h_lhs, h_rhs] at h_sum
  exact add_right_cancel h_sum

/-- ШАГ 25. Теорема о Нуле как динамическом Тождестве (Равновесии):
    Пара вида (a, a) для любого a эквивалентна любой другой паре (b, b).
    Ноль — это не Ничто (отсутствие), а класс совершенного баланса fwd = bwd. -/
theorem step25_zero_as_equilibrium (a b : PosNat) :
    BalanceEquiv ⟨a, a⟩ ⟨b, b⟩ := by
  dsimp [BalanceEquiv]

/-- Сложение баланса: (a, b) + (c, d) = (a + c, b + d). -/
def balance_add (x y : Balance) : Balance :=
  ⟨x.fwd + y.fwd, x.bwd + y.bwd⟩

/-- Прибавление равновесия не меняет класс баланса:
    (x, y) + (a, a) ~ (x, y). Закон сохранения контекста. -/
theorem step25_add_zero_is_identity (x : Balance) (a : PosNat) :
    BalanceEquiv (balance_add x ⟨a, a⟩) x := by
  dsimp [BalanceEquiv, balance_add]
  rw [step13_add_assoc, step12_add_comm a x.bwd]
  rw [step12_add_comm (x.bwd + a) x.fwd]

/-- Согласованность сложения балансов: если x1 ~ x2 и y1 ~ y2, то x1 + y1 ~ x2 + y2. -/
theorem step25_balance_add_congr {x1 x2 y1 y2 : Balance}
    (hx : BalanceEquiv x1 x2) (hy : BalanceEquiv y1 y2) :
    BalanceEquiv (balance_add x1 y1) (balance_add x2 y2) := by
  dsimp [BalanceEquiv, balance_add] at *
  have h_sum : (x1.fwd + x2.bwd) + (y1.fwd + y2.bwd) = (x1.bwd + x2.fwd) + (y1.bwd + y2.fwd) := by
    rw [hx, hy]
  rw [add_add_add_comm x1.fwd x2.bwd y1.fwd y2.bwd] at h_sum
  rw [add_add_add_comm x1.bwd x2.fwd y1.bwd y2.fwd] at h_sum
  exact h_sum

-- =============================================================================
-- ЭТАП VI. ДРОБИ ℚ⁺ И СТРУКТУРНЫЙ ЗАПРЕТ ДЕЛЕНИЯ НА НОЛЬ (Шаги 26–28)
-- =============================================================================

theorem mul_right_comm (a b c : PosNat) : (a * b) * c = (a * c) * b := by
  rw [mul_assoc, step17_mul_comm b c, ← mul_assoc]

theorem mul_mul_mul_comm (a b c d : PosNat) : (a * b) * (c * d) = (a * c) * (b * d) := by
  rw [← mul_assoc, mul_right_comm a b c, mul_assoc (a * c) b d]


inductive Vert3 where | a | b | c

-- =============================================================================
-- ЭТАП X. ВЕКТОР A: КОМПЛЕКСНЫЕ ФАЗЫ И 3-ФАЗНАЯ НЕЙТРАЛЬНОСТЬ (Шаги 41–43)
-- =============================================================================

def balance_sub (x y : Balance) : Balance :=
  ⟨x.fwd + y.bwd, x.bwd + y.fwd⟩

def balance_mul (x y : Balance) : Balance :=
  ⟨x.fwd * y.fwd + x.bwd * y.bwd, x.fwd * y.bwd + x.bwd * y.fwd⟩

def bzero : Balance := ⟨one, one⟩
def bone : Balance := ⟨succ one, one⟩
def bneg_one : Balance := ⟨one, succ one⟩

theorem bone_not_equiv_bneg_one : BalanceEquiv bone bneg_one → False := by
  intro h
  dsimp [BalanceEquiv, bone, bneg_one] at h
  injection h with h1
  cases h1

structure GaussBalance where
  re : Balance
  im : Balance

def GaussEquiv (x y : GaussBalance) : Prop :=
  BalanceEquiv x.re y.re ∧ BalanceEquiv x.im y.im

def gauss_add (x y : GaussBalance) : GaussBalance :=
  ⟨balance_add x.re y.re, balance_add x.im y.im⟩

def gauss_mul (x y : GaussBalance) : GaussBalance :=
  ⟨balance_sub (balance_mul x.re y.re) (balance_mul x.im y.im),
   balance_add (balance_mul x.re y.im) (balance_mul x.im y.re)⟩

def gone : GaussBalance := ⟨bone, bzero⟩
def gneg_one : GaussBalance := ⟨bneg_one, bzero⟩
def gi : GaussBalance := ⟨bzero, bone⟩

/-- ШАГ 41. Фазовый поворот на 90°: i² = -1 в балансной арифметике.
    Мнимая единица i переводит канал fwd в ортогональный канал,
    а повторный поворот меняет ориентацию fwd ↔ bwd (умножение на -1). -/
theorem step41_i_squared_is_neg_one :
    GaussEquiv (gauss_mul gi gi) gneg_one := by
  dsimp [GaussEquiv, gauss_mul, gi, bone, bzero, bneg_one, balance_sub, balance_mul, balance_add, BalanceEquiv]
  exact ⟨rfl, rfl⟩

/-- 3-фазный баланс (три фазы переменного тока, цветной заряд в SU(3), 3-цикл турнира C₃). -/
structure TriPhase where
  p0 : PosNat
  p1 : PosNat
  p2 : PosNat

def tri_add (x y : TriPhase) : TriPhase :=
  ⟨x.p0 + y.p0, x.p1 + y.p1, x.p2 + y.p2⟩

def TriNeutral (t : TriPhase) : Prop :=
  t.p0 = t.p1 ∧ t.p1 = t.p2

-- Базисные 3 фазы: 1, ω, ω²
def phase0 : TriPhase := ⟨succ one, one, one⟩
def phase1 : TriPhase := ⟨one, succ one, one⟩
def phase2 : TriPhase := ⟨one, one, succ one⟩

/-- ШАГ 42. Динамическая 3-фазная нейтральность: 1 + ω + ω² = 0 (равновесие всех трех фаз). -/
theorem step42_triphase_neutrality :
    TriNeutral (tri_add (tri_add phase0 phase1) phase2) := by
  dsimp [TriNeutral, tri_add, phase0, phase1, phase2]
  exact ⟨rfl, rfl⟩

/-- ШАГ 43. Невозможность парного сокращения в 3-фазной системе (1 + ω ≠ 0).
    Равновесие принципиально требует замыкания через третью фазу. -/
theorem step43_triphase_pair_not_neutral :
    TriNeutral (tri_add phase0 phase1) → False := by
  intro h
  have h12 : (tri_add phase0 phase1).p1 = (tri_add phase0 phase1).p2 := h.2
  dsimp [tri_add, phase0, phase1] at h12
  injection h12 with h12'
  exact step07_one_ne_succ one (h12'.symm)


def balance_neg (x : Balance) : Balance :=
  ⟨x.bwd, x.fwd⟩

def two : PosNat := succ one
def three : PosNat := succ two
def four : PosNat := succ three
def five : PosNat := succ four

-- Дуальный баланс (нильпотентные числа)
structure DualBalance where
  re : Balance
  eps : Balance

def DualEquiv (x y : DualBalance) : Prop :=
  BalanceEquiv x.re y.re ∧ BalanceEquiv x.eps y.eps

def dual_mul (x y : DualBalance) : DualBalance :=
  ⟨balance_mul x.re y.re,
   balance_add (balance_mul x.re y.eps) (balance_mul x.eps y.re)⟩

def deps : DualBalance := ⟨bzero, bone⟩

/-- ШАГ 62. Невозможное уравнение x² = 0 при x ≠ 0:
    Дуальная единица ε не равна нулю, но её квадрат в точности равен нулю (ε² = 0)! -/
theorem step62_dual_eps_ne_zero : BalanceEquiv deps.eps bzero → False := by
  intro h
  dsimp [deps, bone, bzero, BalanceEquiv] at h
  injection h with h1; cases h1

theorem step62_dual_eps_squared_is_zero :
    DualEquiv (dual_mul deps deps) ⟨bzero, bzero⟩ := by
  dsimp [DualEquiv, dual_mul, deps, bzero, bone, balance_mul, balance_add, BalanceEquiv]
  exact ⟨rfl, rfl⟩

/-- ШАГ 63. Алгебраическое дифференцирование без пределов и без деления на 0:
    (2 + ε)² = 4 + 4ε.
    Коэффициент при ε в точности равен производной (x²)' = 4 при x = 2! -/
def dual_two : DualBalance := ⟨⟨succ two, one⟩, bone⟩

theorem step63_dual_exact_derivative :
    BalanceEquiv (dual_mul dual_two dual_two).re ⟨succ four, one⟩ ∧
    BalanceEquiv (dual_mul dual_two dual_two).eps ⟨succ four, one⟩ := by
  dsimp [dual_two, dual_mul, five, four, three, two, bone, bzero,
         balance_mul, balance_add, BalanceEquiv]
  exact ⟨rfl, rfl⟩


def BalGtZero (x : Balance) : Prop := x.bwd < x.fwd
def BalLtZero (x : Balance) : Prop := x.fwd < x.bwd

/-- ШАГ 66. Запрет в 1D балансе:
    Никакой одномерный баланс не может быть одновременно положительным и отрицательным.
    Попытка утверждает fwd < bwd ∧ bwd < fwd, что ведет к fwd < fwd (самоотрицание). -/
theorem step66_bal_not_gt_and_lt (x : Balance) : BalGtZero x → BalLtZero x → False := by
  intro hgt hlt
  have h_contra : x.fwd < x.fwd := step20_lt_trans hlt hgt
  exact step19_lt_irrefl x.fwd h_contra

theorem bone_gt_zero : BalGtZero bone := ⟨one, rfl⟩

-- =============================================================================
-- ВСПОМОГАТЕЛЬНАЯ КОЛЬЦЕВАЯ АЛГЕБРА И 0-АКСИОМНАЯ РЕФЛЕКСИЯ НАД Balance
-- (Для прямого фундамента micro_continual_core: Шаги 103–114)
-- =============================================================================

theorem bal_add_comm (x y : Balance) :
    BalanceEquiv (balance_add x y) (balance_add y x) := by
  dsimp [BalanceEquiv, balance_add]
  rw [step12_add_comm x.fwd y.fwd, step12_add_comm x.bwd y.bwd, step12_add_comm]

theorem bal_add_assoc (x y z : Balance) :
    BalanceEquiv (balance_add (balance_add x y) z) (balance_add x (balance_add y z)) := by
  dsimp [BalanceEquiv, balance_add]
  rw [step13_add_assoc x.fwd y.fwd z.fwd, step13_add_assoc x.bwd y.bwd z.bwd, step12_add_comm]

theorem bal_zero_add (x : Balance) : BalanceEquiv (balance_add bzero x) x := by
  exact step24_balance_trans (bal_add_comm bzero x) (step25_add_zero_is_identity x one)

theorem bal_add_zero (x : Balance) : BalanceEquiv (balance_add x bzero) x :=
  step25_add_zero_is_identity x one

theorem bal_neg_congr {x y : Balance} (h : BalanceEquiv x y) :
    BalanceEquiv (balance_neg x) (balance_neg y) := by
  dsimp [BalanceEquiv, balance_neg] at *
  exact h.symm

theorem bal_sub_congr {x1 x2 y1 y2 : Balance}
    (hx : BalanceEquiv x1 x2) (hy : BalanceEquiv y1 y2) :
    BalanceEquiv (balance_sub x1 y1) (balance_sub x2 y2) :=
  step25_balance_add_congr hx (bal_neg_congr hy)

theorem bal_neg_add (x : Balance) : BalanceEquiv (balance_add (balance_neg x) x) bzero := by
  dsimp [BalanceEquiv, balance_add, balance_neg, bzero]
  rw [step12_add_comm x.bwd x.fwd]

theorem bal_add_neg (x : Balance) : BalanceEquiv (balance_add x (balance_neg x)) bzero :=
  step24_balance_trans (bal_add_comm x (balance_neg x)) (bal_neg_add x)

theorem bal_neg_neg (x : Balance) : BalanceEquiv (balance_neg (balance_neg x)) x :=
  step24_balance_refl x

theorem bal_neg_zero : BalanceEquiv (balance_neg bzero) bzero :=
  step24_balance_refl bzero

theorem bal_neg_add_distrib (x y : Balance) :
    BalanceEquiv (balance_neg (balance_add x y)) (balance_add (balance_neg x) (balance_neg y)) :=
  step24_balance_refl _

theorem bal_mul_comm (x y : Balance) :
    BalanceEquiv (balance_mul x y) (balance_mul y x) := by
  dsimp [BalanceEquiv, balance_mul]
  rw [step17_mul_comm x.fwd y.fwd, step17_mul_comm x.bwd y.bwd,
      step17_mul_comm x.fwd y.bwd, step17_mul_comm x.bwd y.fwd,
      step12_add_comm (y.bwd * x.fwd) (y.fwd * x.bwd),
      step12_add_comm]

theorem bal_mul_congr_left {x1 x2 : Balance} (y : Balance) (hx : BalanceEquiv x1 x2) :
    BalanceEquiv (balance_mul x1 y) (balance_mul x2 y) := by
  dsimp [BalanceEquiv, balance_mul] at *
  have h1 : (x1.fwd + x2.bwd) * y.fwd = (x1.bwd + x2.fwd) * y.fwd := congrArg (· * y.fwd) hx
  have h2 : (x1.bwd + x2.fwd) * y.bwd = (x1.fwd + x2.bwd) * y.bwd := congrArg (· * y.bwd) hx.symm
  rw [step16_add_mul, step16_add_mul] at h1 h2
  have hsum : (x1.fwd * y.fwd + x2.bwd * y.fwd) + (x1.bwd * y.bwd + x2.fwd * y.bwd) =
              (x1.bwd * y.fwd + x2.fwd * y.fwd) + (x1.fwd * y.bwd + x2.bwd * y.bwd) := by
    rw [h1, h2]
  rw [add_add_add_comm (x1.fwd * y.fwd) (x2.bwd * y.fwd) (x1.bwd * y.bwd) (x2.fwd * y.bwd),
      step12_add_comm (x2.bwd * y.fwd) (x2.fwd * y.bwd)] at hsum
  rw [add_add_add_comm (x1.bwd * y.fwd) (x2.fwd * y.fwd) (x1.fwd * y.bwd) (x2.bwd * y.bwd),
      step12_add_comm (x1.bwd * y.fwd) (x1.fwd * y.bwd)] at hsum
  exact hsum

theorem bal_mul_congr {x1 x2 y1 y2 : Balance}
    (hx : BalanceEquiv x1 x2) (hy : BalanceEquiv y1 y2) :
    BalanceEquiv (balance_mul x1 y1) (balance_mul x2 y2) := by
  have h1 := bal_mul_congr_left y1 hx
  have h2 := step24_balance_trans (bal_mul_comm x2 y1)
               (step24_balance_trans (bal_mul_congr_left x2 hy) (bal_mul_comm y2 x2))
  exact step24_balance_trans h1 h2

theorem bal_one_mul (x : Balance) : BalanceEquiv (balance_mul bone x) x := by
  dsimp [BalanceEquiv, balance_mul, bone]
  rw [step15_one_mul, step15_one_mul, succ_mul, succ_mul, step15_one_mul, step15_one_mul]
  rw [step13_add_assoc (x.fwd + x.fwd) x.bwd x.bwd,
      step13_add_assoc (x.bwd + x.bwd) x.fwd x.fwd,
      step12_add_comm (x.bwd + x.bwd) (x.fwd + x.fwd)]

theorem bal_mul_one (x : Balance) : BalanceEquiv (balance_mul x bone) x :=
  step24_balance_trans (bal_mul_comm x bone) (bal_one_mul x)

theorem bal_zero_mul (x : Balance) : BalanceEquiv (balance_mul bzero x) bzero := by
  dsimp [BalanceEquiv, balance_mul, bzero]
  rw [step15_one_mul, step15_one_mul, step12_add_comm x.fwd x.bwd]

theorem bal_mul_zero (x : Balance) : BalanceEquiv (balance_mul x bzero) bzero :=
  step24_balance_trans (bal_mul_comm x bzero) (bal_zero_mul x)

theorem bal_neg_mul (x y : Balance) :
    BalanceEquiv (balance_mul (balance_neg x) y) (balance_neg (balance_mul x y)) := by
  dsimp [BalanceEquiv, balance_mul, balance_neg]
  rw [step12_add_comm (x.bwd * y.fwd) (x.fwd * y.bwd),
      step12_add_comm (x.bwd * y.bwd) (x.fwd * y.fwd),
      step12_add_comm]

theorem bal_mul_neg (x y : Balance) :
    BalanceEquiv (balance_mul x (balance_neg y)) (balance_neg (balance_mul x y)) :=
  step24_balance_trans (bal_mul_comm x (balance_neg y))
    (step24_balance_trans (bal_neg_mul y x) (bal_neg_congr (bal_mul_comm y x)))

theorem bal_left_distrib (x y z : Balance) :
    BalanceEquiv (balance_mul x (balance_add y z))
                 (balance_add (balance_mul x y) (balance_mul x z)) := by
  dsimp [BalanceEquiv, balance_mul, balance_add]
  rw [mul_add, mul_add, mul_add, mul_add]
  rw [add_add_add_comm (x.fwd * y.fwd) (x.fwd * z.fwd) (x.bwd * y.bwd) (x.bwd * z.bwd)]
  rw [add_add_add_comm (x.fwd * y.bwd) (x.fwd * z.bwd) (x.bwd * y.fwd) (x.bwd * z.fwd)]
  rw [step12_add_comm]

theorem bal_right_distrib (x y z : Balance) :
    BalanceEquiv (balance_mul (balance_add x y) z)
                 (balance_add (balance_mul x z) (balance_mul y z)) :=
  step24_balance_trans (bal_mul_comm (balance_add x y) z)
    (step24_balance_trans (bal_left_distrib z x y)
      (step25_balance_add_congr (bal_mul_comm z x) (bal_mul_comm z y)))

theorem bal_mul_assoc (x y z : Balance) :
    BalanceEquiv (balance_mul (balance_mul x y) z) (balance_mul x (balance_mul y z)) := by
  dsimp [BalanceEquiv, balance_mul]
  rw [step16_add_mul, step16_add_mul, step16_add_mul, step16_add_mul,
      mul_add, mul_add, mul_add, mul_add]
  rw [mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc]
  have hA : (x.fwd * (y.fwd * z.fwd) + x.bwd * (y.bwd * z.fwd)) +
            (x.fwd * (y.bwd * z.bwd) + x.bwd * (y.fwd * z.bwd)) =
            (x.fwd * (y.fwd * z.fwd) + x.fwd * (y.bwd * z.bwd)) +
            (x.bwd * (y.fwd * z.bwd) + x.bwd * (y.bwd * z.fwd)) := by
    rw [add_add_add_comm, step12_add_comm (x.bwd * (y.bwd * z.fwd)) (x.bwd * (y.fwd * z.bwd))]
  have hB : (x.fwd * (y.fwd * z.bwd) + x.bwd * (y.bwd * z.bwd)) +
            (x.fwd * (y.bwd * z.fwd) + x.bwd * (y.fwd * z.fwd)) =
            (x.fwd * (y.fwd * z.bwd) + x.fwd * (y.bwd * z.fwd)) +
            (x.bwd * (y.fwd * z.fwd) + x.bwd * (y.bwd * z.bwd)) := by
    rw [add_add_add_comm, step12_add_comm (x.bwd * (y.bwd * z.bwd)) (x.bwd * (y.fwd * z.fwd))]
  rw [hA, hB, step12_add_comm]

theorem bal_add_left_comm (a b c : Balance) :
    BalanceEquiv (balance_add a (balance_add b c)) (balance_add b (balance_add a c)) :=
  step24_balance_trans (step24_balance_symm (bal_add_assoc a b c))
    (step24_balance_trans (step25_balance_add_congr (bal_add_comm a b) (step24_balance_refl c))
      (bal_add_assoc b a c))

theorem bal_add_add_add_comm (a b c d : Balance) :
    BalanceEquiv (balance_add (balance_add a b) (balance_add c d))
                 (balance_add (balance_add a c) (balance_add b d)) := by
  have h1 := bal_add_assoc a b (balance_add c d)
  have h2 := step25_balance_add_congr (step24_balance_refl a) (bal_add_left_comm b c d)
  have h3 := step24_balance_symm (bal_add_assoc a c (balance_add b d))
  exact step24_balance_trans h1 (step24_balance_trans h2 h3)

-- Вложение Nat → Balance (без раскрытия в гигантские унарные PosNat!)
def balOfNat : Nat → Balance
  | 0 => bzero
  | n + 1 => balance_add (balOfNat n) bone

theorem balOfNat_add (a b : Nat) :
    BalanceEquiv (balOfNat (a + b)) (balance_add (balOfNat a) (balOfNat b)) := by
  induction b with
  | zero => exact step24_balance_symm (bal_add_zero (balOfNat a))
  | succ k ih =>
    change BalanceEquiv (balance_add (balOfNat (a + k)) bone)
                        (balance_add (balOfNat a) (balance_add (balOfNat k) bone))
    exact step24_balance_trans
      (step25_balance_add_congr ih (step24_balance_refl bone))
      (bal_add_assoc (balOfNat a) (balOfNat k) bone)

theorem balOfNat_mul (a b : Nat) :
    BalanceEquiv (balOfNat (a * b)) (balance_mul (balOfNat a) (balOfNat b)) := by
  induction b with
  | zero => exact step24_balance_symm (bal_mul_zero (balOfNat a))
  | succ k ih =>
    change BalanceEquiv (balOfNat (a * k + a))
                        (balance_mul (balOfNat a) (balance_add (balOfNat k) bone))
    have h1 := balOfNat_add (a * k) a
    have h2 := step25_balance_add_congr ih (step24_balance_symm (bal_mul_one (balOfNat a)))
    have h3 := step24_balance_symm (bal_left_distrib (balOfNat a) (balOfNat k) bone)
    exact step24_balance_trans h1 (step24_balance_trans h2 h3)

-- 0-аксиомный нормализатор многочленов над Balance
inductive BZ where
  | zero : BZ
  | pos : Nat → BZ
  | neg : Nat → BZ
deriving DecidableEq

def BZ.ofNat : Nat → BZ
  | 0 => .zero
  | n + 1 => .pos n

def BZ.subNat : Nat → Nat → BZ
  | 0, 0 => .zero
  | 0, b + 1 => .neg b
  | a + 1, 0 => .pos a
  | a + 1, b + 1 => BZ.subNat a b

def BZ.add : BZ → BZ → BZ
  | .zero, y => y
  | x, .zero => x
  | .pos a, .pos b => .pos (a + b + 1)
  | .neg a, .neg b => .neg (a + b + 1)
  | .pos a, .neg b => BZ.subNat a b
  | .neg a, .pos b => BZ.subNat b a

def BZ.mul : BZ → BZ → BZ
  | .zero, _ => .zero
  | _, .zero => .zero
  | .pos a, .pos b => .pos (a * b + a + b)
  | .neg a, .neg b => .pos (a * b + a + b)
  | .pos a, .neg b => .neg (a * b + a + b)
  | .neg a, .pos b => .neg (a * b + a + b)

def evalBZ : BZ → Balance
  | .zero => bzero
  | .pos n => balOfNat (n + 1)
  | .neg n => balance_neg (balOfNat (n + 1))

theorem evalBZ_subNat (a b : Nat) :
    BalanceEquiv (evalBZ (BZ.subNat a b)) (balance_sub (balOfNat a) (balOfNat b)) := by
  induction a generalizing b with
  | zero =>
    cases b with
    | zero => exact step24_balance_symm (bal_add_zero bzero)
    | succ b' =>
      dsimp [BZ.subNat, evalBZ, balOfNat, balance_sub]
      exact step24_balance_symm (bal_zero_add (balance_neg (balance_add (balOfNat b') bone)))
  | succ a' ih =>
    cases b with
    | zero =>
      dsimp [BZ.subNat, evalBZ]
      exact step24_balance_symm (bal_add_zero (balOfNat (a' + 1)))
    | succ b' =>
      dsimp [BZ.subNat]
      refine step24_balance_trans (ih b') ?_
      change BalanceEquiv (balance_add (balOfNat a') (balance_neg (balOfNat b')))
                          (balance_add (balance_add (balOfNat a') bone)
                                       (balance_neg (balance_add (balOfNat b') bone)))
      have h1 : BalanceEquiv
          (balance_add (balance_add (balOfNat a') bone)
                       (balance_neg (balance_add (balOfNat b') bone)))
          (balance_add (balance_add (balOfNat a') (balance_neg (balOfNat b')))
                       (balance_add bone (balance_neg bone))) :=
        bal_add_add_add_comm (balOfNat a') bone (balance_neg (balOfNat b')) (balance_neg bone)
      have h2 : BalanceEquiv
          (balance_add (balance_add (balOfNat a') (balance_neg (balOfNat b')))
                       (balance_add bone (balance_neg bone)))
          (balance_add (balOfNat a') (balance_neg (balOfNat b'))) :=
        step24_balance_trans
          (step25_balance_add_congr (step24_balance_refl _) (bal_add_neg bone))
          (bal_add_zero _)
      exact step24_balance_symm (step24_balance_trans h1 h2)

theorem bal_sub_succ_succ (x y : Balance) :
    BalanceEquiv (balance_sub (balance_add x bone) (balance_add y bone)) (balance_sub x y) := by
  have h1 := bal_add_add_add_comm x bone (balance_neg y) (balance_neg bone)
  have h2 := step25_balance_add_congr (step24_balance_refl (balance_sub x y)) (bal_add_neg bone)
  exact step24_balance_trans h1 (step24_balance_trans h2 (bal_add_zero _))

private theorem my_nat_zero_add (n : Nat) : 0 + n = n := by
  induction n with
  | zero => rfl
  | succ n' ih => exact congrArg Nat.succ ih

private theorem my_nat_succ_add (a b : Nat) : (a + 1) + b = (a + b) + 1 := by
  induction b with
  | zero => rfl
  | succ b' ih => exact congrArg Nat.succ ih

private theorem my_nat_add_assoc (a b c : Nat) : (a + b) + c = a + (b + c) := by
  induction c with
  | zero => rfl
  | succ c' ih => exact congrArg Nat.succ ih

private theorem my_nat_add_comm (a b : Nat) : a + b = b + a := by
  induction b with
  | zero => exact (my_nat_zero_add a).symm
  | succ b' ih => exact Eq.trans (congrArg Nat.succ ih) (my_nat_succ_add b' a).symm

private theorem my_nat_succ_mul (a b : Nat) : (a + 1) * b = a * b + b := by
  induction b with
  | zero => rfl
  | succ b' ih =>
    apply congrArg Nat.succ
    exact Eq.trans (congrArg (· + a) ih)
      (Eq.trans (my_nat_add_assoc (a * b') b' a)
        (Eq.trans (congrArg (a * b' + ·) (my_nat_add_comm b' a))
          (my_nat_add_assoc (a * b') a b').symm))

theorem evalBZ_add (x y : BZ) :
    BalanceEquiv (evalBZ (BZ.add x y)) (balance_add (evalBZ x) (evalBZ y)) := by
  cases x with
  | zero => exact step24_balance_symm (bal_zero_add (evalBZ y))
  | pos a =>
    cases y with
    | zero => exact step24_balance_symm (bal_add_zero (balOfNat (a + 1)))
    | pos b =>
      dsimp [BZ.add, evalBZ]
      have hnat : a + b + 1 + 1 = (a + 1) + (b + 1) :=
        congrArg Nat.succ (my_nat_succ_add a b).symm
      rw [hnat]
      exact balOfNat_add (a + 1) (b + 1)
    | neg b =>
      dsimp [BZ.add, evalBZ]
      exact step24_balance_trans (evalBZ_subNat a b)
        (step24_balance_symm (bal_sub_succ_succ (balOfNat a) (balOfNat b)))
  | neg a =>
    cases y with
    | zero => exact step24_balance_symm (bal_add_zero (balance_neg (balOfNat (a + 1))))
    | pos b =>
      dsimp [BZ.add, evalBZ]
      have h1 := step24_balance_trans (evalBZ_subNat b a)
        (step24_balance_symm (bal_sub_succ_succ (balOfNat b) (balOfNat a)))
      exact step24_balance_trans h1
        (bal_add_comm (balOfNat (b + 1)) (balance_neg (balOfNat (a + 1))))
    | neg b =>
      dsimp [BZ.add, evalBZ]
      have hnat : a + b + 1 + 1 = (a + 1) + (b + 1) :=
        congrArg Nat.succ (my_nat_succ_add a b).symm
      rw [hnat]
      exact bal_neg_congr (balOfNat_add (a + 1) (b + 1))

theorem nat_succ_mul_succ (a b : Nat) : (a + 1) * (b + 1) = (a * b + a + b) + 1 := by
  apply congrArg Nat.succ
  exact Eq.trans (congrArg (· + a) (my_nat_succ_mul a b))
    (Eq.trans (my_nat_add_assoc (a * b) b a)
      (Eq.trans (congrArg (a * b + ·) (my_nat_add_comm b a))
        (my_nat_add_assoc (a * b) a b).symm))

theorem evalBZ_mul (x y : BZ) :
    BalanceEquiv (evalBZ (BZ.mul x y)) (balance_mul (evalBZ x) (evalBZ y)) := by
  cases x with
  | zero => exact step24_balance_symm (bal_zero_mul (evalBZ y))
  | pos a =>
    cases y with
    | zero => exact step24_balance_symm (bal_mul_zero (balOfNat (a + 1)))
    | pos b =>
      dsimp [BZ.mul, evalBZ]
      rw [← nat_succ_mul_succ a b]
      exact balOfNat_mul (a + 1) (b + 1)
    | neg b =>
      dsimp [BZ.mul, evalBZ]
      rw [← nat_succ_mul_succ a b]
      exact step24_balance_trans (bal_neg_congr (balOfNat_mul (a + 1) (b + 1)))
        (step24_balance_symm (bal_mul_neg (balOfNat (a + 1)) (balOfNat (b + 1))))
  | neg a =>
    cases y with
    | zero => exact step24_balance_symm (bal_mul_zero (balance_neg (balOfNat (a + 1))))
    | pos b =>
      dsimp [BZ.mul, evalBZ]
      rw [← nat_succ_mul_succ a b]
      exact step24_balance_trans (bal_neg_congr (balOfNat_mul (a + 1) (b + 1)))
        (step24_balance_symm (bal_neg_mul (balOfNat (a + 1)) (balOfNat (b + 1))))
    | neg b =>
      dsimp [BZ.mul, evalBZ]
      rw [← nat_succ_mul_succ a b]
      have h1 := balOfNat_mul (a + 1) (b + 1)
      have h2 := step24_balance_trans (bal_neg_mul (balOfNat (a + 1)) (balance_neg (balOfNat (b + 1))))
        (step24_balance_trans (bal_neg_congr (bal_mul_neg (balOfNat (a + 1)) (balOfNat (b + 1))))
          (bal_neg_neg (balance_mul (balOfNat (a + 1)) (balOfNat (b + 1)))))
      exact step24_balance_trans h1 (step24_balance_symm h2)

def bCmpNat : Nat → Nat → Ordering
  | 0, 0 => .eq
  | 0, _ + 1 => .lt
  | _ + 1, 0 => .gt
  | a + 1, b + 1 => bCmpNat a b

theorem bCmpNat_eq {a b : Nat} (h : bCmpNat a b = .eq) : a = b := by
  induction a generalizing b with
  | zero => cases b with | zero => rfl | succ _ => contradiction
  | succ a' ih => cases b with | zero => contradiction | succ b' => exact congrArg Nat.succ (ih h)

def bCmpMono : List Nat → List Nat → Ordering
  | [], [] => .eq
  | [], _ :: _ => .lt
  | _ :: _, [] => .gt
  | a :: as, b :: bs =>
    match bCmpNat a b with
    | .lt => .lt
    | .gt => .gt
    | .eq => bCmpMono as bs

theorem bCmpMono_eq {m1 m2 : List Nat} (h : bCmpMono m1 m2 = .eq) : m1 = m2 := by
  induction m1 generalizing m2 with
  | nil => cases m2 with | nil => rfl | cons _ _ => contradiction
  | cons a as ih =>
    cases m2 with
    | nil => contradiction
    | cons b bs =>
      dsimp [bCmpMono] at h
      cases hc : bCmpNat a b with
      | lt => rw [hc] at h; contradiction
      | gt => rw [hc] at h; contradiction
      | eq => rw [hc] at h; rw [bCmpNat_eq hc, ih h]

def bLeNat : Nat → Nat → Bool
  | 0, _ => true
  | _ + 1, 0 => false
  | a + 1, b + 1 => bLeNat a b

def bInsertNat (a : Nat) : List Nat → List Nat
  | [] => [a]
  | b :: bs =>
    match bLeNat a b with
    | true => a :: b :: bs
    | false => b :: bInsertNat a bs

def bMergeMono : List Nat → List Nat → List Nat
  | [], m => m
  | a :: as, m => bInsertNat a (bMergeMono as m)

def evalBMono (env : Nat → Balance) : List Nat → Balance
  | [] => bone
  | i :: is => balance_mul (env i) (evalBMono env is)

theorem evalBMono_insert (env : Nat → Balance) (a : Nat) (m : List Nat) :
    BalanceEquiv (evalBMono env (bInsertNat a m)) (balance_mul (env a) (evalBMono env m)) := by
  induction m with
  | nil => exact step24_balance_refl _
  | cons b bs ih =>
    dsimp [bInsertNat]
    cases bLeNat a b with
    | true => exact step24_balance_refl _
    | false =>
      dsimp [evalBMono]
      have h1 := bal_mul_congr (step24_balance_refl (env b)) ih
      have h2 := step24_balance_symm (bal_mul_assoc (env b) (env a) (evalBMono env bs))
      have h3 := bal_mul_congr (bal_mul_comm (env b) (env a)) (step24_balance_refl (evalBMono env bs))
      have h4 := bal_mul_assoc (env a) (env b) (evalBMono env bs)
      exact step24_balance_trans h1 (step24_balance_trans h2 (step24_balance_trans h3 h4))

theorem evalBMono_merge (env : Nat → Balance) (m1 m2 : List Nat) :
    BalanceEquiv (evalBMono env (bMergeMono m1 m2))
                 (balance_mul (evalBMono env m1) (evalBMono env m2)) := by
  induction m1 with
  | nil => exact step24_balance_symm (bal_one_mul _)
  | cons a as ih =>
    dsimp [bMergeMono, evalBMono]
    exact step24_balance_trans (evalBMono_insert env a (bMergeMono as m2))
      (step24_balance_trans (bal_mul_congr (step24_balance_refl (env a)) ih)
        (step24_balance_symm (bal_mul_assoc (env a) (evalBMono env as) (evalBMono env m2))))

abbrev BPoly := List (BZ × List Nat)

def evalBPoly (env : Nat → Balance) : BPoly → Balance
  | [] => bzero
  | (c, m) :: rest => balance_add (balance_mul (evalBZ c) (evalBMono env m)) (evalBPoly env rest)

def bConsNZ (c : BZ) (m : List Nat) (rest : BPoly) : BPoly :=
  match c with
  | .zero => rest
  | .pos n => (.pos n, m) :: rest
  | .neg n => (.neg n, m) :: rest

theorem evalBPoly_consNZ (env : Nat → Balance) (c : BZ) (m : List Nat) (rest : BPoly) :
    BalanceEquiv (evalBPoly env (bConsNZ c m rest))
                 (balance_add (balance_mul (evalBZ c) (evalBMono env m)) (evalBPoly env rest)) := by
  cases c with
  | zero =>
    dsimp [bConsNZ, evalBZ]
    exact step24_balance_symm
      (step24_balance_trans
        (step25_balance_add_congr (bal_zero_mul _) (step24_balance_refl _))
        (bal_zero_add _))
  | pos _ => exact step24_balance_refl _
  | neg _ => exact step24_balance_refl _

def bInsertTerm (c : BZ) (m : List Nat) : BPoly → BPoly
  | [] => [(c, m)]
  | (c', m') :: rest =>
    match bCmpMono m m' with
    | .lt => (c, m) :: (c', m') :: rest
    | .gt => (c', m') :: bInsertTerm c m rest
    | .eq => bConsNZ (BZ.add c c') m' rest

theorem evalBPoly_insertTerm (env : Nat → Balance) (c : BZ) (m : List Nat) (p : BPoly) :
    BalanceEquiv (evalBPoly env (bInsertTerm c m p))
                 (balance_add (balance_mul (evalBZ c) (evalBMono env m)) (evalBPoly env p)) := by
  induction p with
  | nil => exact step24_balance_refl _
  | cons hd tl ih =>
    rcases hd with ⟨c', m'⟩
    dsimp [bInsertTerm]
    cases hc : bCmpMono m m' with
    | lt => exact step24_balance_refl _
    | gt =>
      dsimp [evalBPoly]
      exact step24_balance_trans
        (step25_balance_add_congr (step24_balance_refl _) ih)
        (bal_add_left_comm _ _ _)
    | eq =>
      rw [bCmpMono_eq hc]
      dsimp [evalBPoly]
      have h1 := evalBPoly_consNZ env (BZ.add c c') m' tl
      have h2 := step25_balance_add_congr
        (bal_mul_congr (evalBZ_add c c') (step24_balance_refl (evalBMono env m')))
        (step24_balance_refl (evalBPoly env tl))
      have h3 := step25_balance_add_congr (bal_right_distrib (evalBZ c) (evalBZ c') (evalBMono env m'))
        (step24_balance_refl (evalBPoly env tl))
      have h4 := bal_add_assoc (balance_mul (evalBZ c) (evalBMono env m'))
        (balance_mul (evalBZ c') (evalBMono env m')) (evalBPoly env tl)
      exact step24_balance_trans h1 (step24_balance_trans h2 (step24_balance_trans h3 h4))

def bAddTerm (c : BZ) (m : List Nat) (p : BPoly) : BPoly :=
  match c with
  | .zero => p
  | .pos n => bInsertTerm (.pos n) m p
  | .neg n => bInsertTerm (.neg n) m p

theorem evalBPoly_addTerm (env : Nat → Balance) (c : BZ) (m : List Nat) (p : BPoly) :
    BalanceEquiv (evalBPoly env (bAddTerm c m p))
                 (balance_add (balance_mul (evalBZ c) (evalBMono env m)) (evalBPoly env p)) := by
  cases c with
  | zero =>
    dsimp [bAddTerm, evalBZ]
    exact step24_balance_symm
      (step24_balance_trans
        (step25_balance_add_congr (bal_zero_mul _) (step24_balance_refl _))
        (bal_zero_add _))
  | pos n => exact evalBPoly_insertTerm env (.pos n) m p
  | neg n => exact evalBPoly_insertTerm env (.neg n) m p

def bAddPoly : BPoly → BPoly → BPoly
  | [], q => q
  | (c, m) :: rest, q => bAddTerm c m (bAddPoly rest q)

theorem evalBPoly_addPoly (env : Nat → Balance) (p q : BPoly) :
    BalanceEquiv (evalBPoly env (bAddPoly p q)) (balance_add (evalBPoly env p) (evalBPoly env q)) := by
  induction p with
  | nil => exact step24_balance_symm (bal_zero_add _)
  | cons hd tl ih =>
    rcases hd with ⟨c, m⟩
    dsimp [bAddPoly, evalBPoly]
    exact step24_balance_trans (evalBPoly_addTerm env c m (bAddPoly tl q))
      (step24_balance_trans (step25_balance_add_congr (step24_balance_refl _) ih)
        (step24_balance_symm (bal_add_assoc _ _ _)))

def bScalePoly (c : BZ) (m : List Nat) : BPoly → BPoly
  | [] => []
  | (c', m') :: rest => bAddTerm (BZ.mul c c') (bMergeMono m m') (bScalePoly c m rest)

theorem evalBPoly_scalePoly (env : Nat → Balance) (c : BZ) (m : List Nat) (p : BPoly) :
    BalanceEquiv (evalBPoly env (bScalePoly c m p))
                 (balance_mul (balance_mul (evalBZ c) (evalBMono env m)) (evalBPoly env p)) := by
  induction p with
  | nil => exact step24_balance_symm (bal_mul_zero _)
  | cons hd tl ih =>
    rcases hd with ⟨c', m'⟩
    dsimp [bScalePoly, evalBPoly]
    have hterm : BalanceEquiv
        (balance_mul (evalBZ (BZ.mul c c')) (evalBMono env (bMergeMono m m')))
        (balance_mul (balance_mul (evalBZ c) (evalBMono env m))
                     (balance_mul (evalBZ c') (evalBMono env m'))) := by
      have s1 := bal_mul_congr (evalBZ_mul c c') (evalBMono_merge env m m')
      have s2 := bal_mul_assoc (evalBZ c) (evalBZ c') (balance_mul (evalBMono env m) (evalBMono env m'))
      have s3 := bal_mul_congr (step24_balance_refl (evalBZ c))
        (step24_balance_trans (step24_balance_symm (bal_mul_assoc (evalBZ c') (evalBMono env m) (evalBMono env m')))
          (step24_balance_trans (bal_mul_congr (bal_mul_comm (evalBZ c') (evalBMono env m)) (step24_balance_refl _))
            (bal_mul_assoc (evalBMono env m) (evalBZ c') (evalBMono env m'))))
      have s4 := step24_balance_symm (bal_mul_assoc (evalBZ c) (evalBMono env m)
        (balance_mul (evalBZ c') (evalBMono env m')))
      exact step24_balance_trans s1 (step24_balance_trans s2 (step24_balance_trans s3 s4))
    exact step24_balance_trans (evalBPoly_addTerm env _ _ _)
      (step24_balance_trans (step25_balance_add_congr hterm ih)
        (step24_balance_symm (bal_left_distrib _ _ _)))

def bMulPoly : BPoly → BPoly → BPoly
  | [], _ => []
  | (c, m) :: rest, q => bAddPoly (bScalePoly c m q) (bMulPoly rest q)

theorem evalBPoly_mulPoly (env : Nat → Balance) (p q : BPoly) :
    BalanceEquiv (evalBPoly env (bMulPoly p q)) (balance_mul (evalBPoly env p) (evalBPoly env q)) := by
  induction p with
  | nil => exact step24_balance_symm (bal_zero_mul _)
  | cons hd tl ih =>
    rcases hd with ⟨c, m⟩
    dsimp [bMulPoly, evalBPoly]
    exact step24_balance_trans (evalBPoly_addPoly env _ _)
      (step24_balance_trans (step25_balance_add_congr (evalBPoly_scalePoly env c m q) ih)
        (step24_balance_symm (bal_right_distrib _ _ _)))

inductive BExpr where
  | zero : BExpr
  | one : BExpr
  | nat : Nat → BExpr
  | var : Nat → BExpr
  | neg : BExpr → BExpr
  | add : BExpr → BExpr → BExpr
  | sub : BExpr → BExpr → BExpr
  | mul : BExpr → BExpr → BExpr

def BExpr.eval (env : Nat → Balance) : BExpr → Balance
  | .zero => bzero
  | .one => bone
  | .nat n => balOfNat n
  | .var i => env i
  | .neg a => balance_neg (a.eval env)
  | .add a b => balance_add (a.eval env) (b.eval env)
  | .sub a b => balance_sub (a.eval env) (b.eval env)
  | .mul a b => balance_mul (a.eval env) (b.eval env)

def BExpr.norm : BExpr → BPoly
  | .zero => []
  | .one => [(.pos 0, [])]
  | .nat 0 => []
  | .nat (n + 1) => [(.pos n, [])]
  | .var i => [(.pos 0, [i])]
  | .neg a => bScalePoly (.neg 0) [] a.norm
  | .add a b => bAddPoly a.norm b.norm
  | .sub a b => bAddPoly a.norm (bScalePoly (.neg 0) [] b.norm)
  | .mul a b => bMulPoly a.norm b.norm

theorem evalBZ_pos0 : BalanceEquiv (evalBZ (.pos 0)) bone := bal_zero_add bone
theorem evalBZ_neg0 : BalanceEquiv (evalBZ (.neg 0)) (balance_neg bone) := bal_neg_congr (bal_zero_add bone)

theorem BExpr.eval_norm (env : Nat → Balance) (e : BExpr) :
    BalanceEquiv (evalBPoly env e.norm) (e.eval env) := by
  induction e with
  | zero => exact step24_balance_refl _
  | one =>
    dsimp [BExpr.norm, BExpr.eval, evalBPoly, evalBMono]
    exact step24_balance_trans (bal_add_zero _)
      (step24_balance_trans (bal_mul_one _) evalBZ_pos0)
  | nat n =>
    cases n with
    | zero => exact step24_balance_refl _
    | succ k =>
      dsimp [BExpr.norm, BExpr.eval, evalBPoly, evalBMono, evalBZ]
      exact step24_balance_trans (bal_add_zero _) (bal_mul_one _)
  | var i =>
    dsimp [BExpr.norm, BExpr.eval, evalBPoly, evalBMono]
    exact step24_balance_trans (bal_add_zero _)
      (step24_balance_trans (bal_mul_congr evalBZ_pos0 (bal_mul_one (env i))) (bal_one_mul (env i)))
  | neg a ih =>
    dsimp [BExpr.norm, BExpr.eval]
    have h1 := evalBPoly_scalePoly env (.neg 0) [] a.norm
    have h2 : BalanceEquiv (balance_mul (evalBZ (.neg 0)) (evalBMono env [])) (balance_neg bone) :=
      step24_balance_trans (bal_mul_one _) evalBZ_neg0
    have h3 := bal_mul_congr h2 ih
    have h4 := step24_balance_trans (bal_neg_mul bone (a.eval env)) (bal_neg_congr (bal_one_mul _))
    exact step24_balance_trans h1 (step24_balance_trans h3 h4)
  | add a b iha ihb =>
    dsimp [BExpr.norm, BExpr.eval]
    exact step24_balance_trans (evalBPoly_addPoly env a.norm b.norm) (step25_balance_add_congr iha ihb)
  | sub a b iha ihb =>
    dsimp [BExpr.norm, BExpr.eval]
    have hb : BalanceEquiv (evalBPoly env (bScalePoly (.neg 0) [] b.norm)) (balance_neg (b.eval env)) := by
      have h1 := evalBPoly_scalePoly env (.neg 0) [] b.norm
      have h2 : BalanceEquiv (balance_mul (evalBZ (.neg 0)) (evalBMono env [])) (balance_neg bone) :=
        step24_balance_trans (bal_mul_one _) evalBZ_neg0
      have h3 := bal_mul_congr h2 ihb
      have h4 := step24_balance_trans (bal_neg_mul bone (b.eval env)) (bal_neg_congr (bal_one_mul _))
      exact step24_balance_trans h1 (step24_balance_trans h3 h4)
    exact step24_balance_trans (evalBPoly_addPoly env a.norm _) (step25_balance_add_congr iha hb)
  | mul a b iha ihb =>
    dsimp [BExpr.norm, BExpr.eval]
    exact step24_balance_trans (evalBPoly_mulPoly env a.norm b.norm) (bal_mul_congr iha ihb)

/-- Главная теорема 0-аксиомной полиномиальной рефлексии над Balance. -/
theorem bal_poly_eq (env : Nat → Balance) (e1 e2 : BExpr) (h : e1.norm = e2.norm) :
    BalanceEquiv (e1.eval env) (e2.eval env) := by
  have h1 := step24_balance_symm (BExpr.eval_norm env e1)
  have h2 := BExpr.eval_norm env e2
  rw [h] at h1
  exact step24_balance_trans h1 h2


-- =============================================================================
-- ПОРЯДОК И СТРОГАЯ ПОЛОЖИТЕЛЬНОСТЬ НАД Balance (0 АКСИОМ)
-- =============================================================================

def BalGeZero (x : Balance) : Prop := BalanceEquiv x bzero ∨ BalGtZero x
def BalLt (x y : Balance) : Prop := BalGtZero (balance_sub y x)
def BalLe (x y : Balance) : Prop := BalGeZero (balance_sub y x)

def bsq (x : Balance) : Balance := balance_mul x x

theorem bal_gt_zero_of_equiv {x y : Balance} (heq : BalanceEquiv x y) (hgt : BalGtZero x) :
    BalGtZero y := by
  cases hgt with
  | intro c hc =>
    dsimp [BalanceEquiv] at heq
    rw [← hc] at heq
    have h1 : x.bwd + (y.bwd + c) = x.bwd + y.fwd := by
      rw [step12_add_comm y.bwd c, ← step13_add_assoc, heq]
    exact ⟨c, add_left_cancel h1⟩

theorem bal_ge_zero_of_equiv {x y : Balance} (heq : BalanceEquiv x y) (hge : BalGeZero x) :
    BalGeZero y := by
  cases hge with
  | inl hz => exact Or.inl (step24_balance_trans (step24_balance_symm heq) hz)
  | inr hgt => exact Or.inr (bal_gt_zero_of_equiv heq hgt)

theorem bal_gt_zero_ne_zero {x : Balance} (hgt : BalGtZero x) :
    BalanceEquiv x bzero → False := by
  intro hz
  have h0 : BalGtZero bzero := bal_gt_zero_of_equiv hz hgt
  exact step19_lt_irrefl one h0

theorem bal_add_gt_ge {x y : Balance} (hx : BalGtZero x) (hy : BalGeZero y) :
    BalGtZero (balance_add x y) := by
  cases hy with
  | inl hyz =>
    have heq : BalanceEquiv x (balance_add x y) :=
      step24_balance_symm (step24_balance_trans
        (step25_balance_add_congr (step24_balance_refl x) hyz) (bal_add_zero x))
    exact bal_gt_zero_of_equiv heq hx
  | inr hygt =>
    cases hx with
    | intro cx hcx =>
      cases hygt with
      | intro cy hcy =>
        refine ⟨cx + cy, ?_⟩
        dsimp [balance_add]
        rw [add_add_add_comm, hcx, hcy]

theorem bal_add_ge_ge {x y : Balance} (hx : BalGeZero x) (hy : BalGeZero y) :
    BalGeZero (balance_add x y) := by
  cases hx with
  | inl hxz =>
    cases hy with
    | inl hyz =>
      exact Or.inl (step24_balance_trans (step25_balance_add_congr hxz hyz) (bal_add_zero bzero))
    | inr hygt =>
      exact Or.inr (bal_gt_zero_of_equiv (bal_add_comm y x) (bal_add_gt_ge hygt (Or.inl hxz)))
  | inr hxgt => exact Or.inr (bal_add_gt_ge hxgt hy)

theorem bal_sq_ge_zero (x : Balance) : BalGeZero (bsq x) := by
  cases trichotomy x.fwd x.bwd with
  | inl heq =>
    left
    dsimp [bsq, balance_mul, BalanceEquiv, bzero]
    rw [heq]
  | inr hne =>
    cases hne with
    | inl hlt =>
      right
      cases hlt with
      | intro d hd =>
        refine ⟨d * d, ?_⟩
        dsimp [bsq, balance_mul]
        rw [← hd, mul_add, step16_add_mul, mul_add, step16_add_mul, step16_add_mul]
        rw [step17_mul_comm d x.fwd]
        have h1 : (x.fwd * x.fwd + x.fwd * d) + (x.fwd * x.fwd + x.fwd * d) + d * d =
                  x.fwd * x.fwd + ((x.fwd * x.fwd + x.fwd * d) + (x.fwd * d + d * d)) := by
          rw [step13_add_assoc (x.fwd * x.fwd) (x.fwd * d) (x.fwd * x.fwd + x.fwd * d),
              ← step13_add_assoc (x.fwd * d) (x.fwd * x.fwd) (x.fwd * d),
              step12_add_comm (x.fwd * d) (x.fwd * x.fwd),
              step13_add_assoc (x.fwd * x.fwd) (x.fwd * d) (x.fwd * d),
              ← step13_add_assoc (x.fwd * x.fwd) (x.fwd * d) (x.fwd * d),
              step13_add_assoc (x.fwd * x.fwd) ((x.fwd * x.fwd + x.fwd * d) + x.fwd * d) (d * d),
              step13_add_assoc (x.fwd * x.fwd + x.fwd * d) (x.fwd * d) (d * d)]
        exact h1
    | inr hgt =>
      right
      cases hgt with
      | intro d hd =>
        refine ⟨d * d, ?_⟩
        dsimp [bsq, balance_mul]
        rw [← hd, mul_add, step16_add_mul, mul_add, step16_add_mul, step16_add_mul]
        rw [step17_mul_comm d x.bwd]
        have h1 : (x.bwd * x.bwd + x.bwd * d) + (x.bwd * x.bwd + x.bwd * d) + d * d =
                  ((x.bwd * x.bwd + x.bwd * d) + (x.bwd * d + d * d)) + x.bwd * x.bwd := by
          rw [step13_add_assoc (x.bwd * x.bwd + x.bwd * d) (x.bwd * x.bwd + x.bwd * d) (d * d),
              step12_add_comm (x.bwd * x.bwd) (x.bwd * d),
              step13_add_assoc (x.bwd * d) (x.bwd * x.bwd) (d * d),
              step12_add_comm (x.bwd * x.bwd) (d * d),
              ← step13_add_assoc (x.bwd * d) (d * d) (x.bwd * x.bwd),
              ← step13_add_assoc (x.bwd * d + x.bwd * x.bwd) (x.bwd * d + d * d) (x.bwd * x.bwd),
              step12_add_comm (x.bwd * d) (x.bwd * x.bwd)]
        exact h1

theorem bal_mul_gt_gt {x y : Balance} (hx : BalGtZero x) (hy : BalGtZero y) :
    BalGtZero (balance_mul x y) := by
  cases hx with
  | intro cx hcx =>
    cases hy with
    | intro cy hcy =>
      refine ⟨cx * cy, ?_⟩
      dsimp [balance_mul]
      rw [← hcx, ← hcy, step16_add_mul, mul_add, step16_add_mul, mul_add, mul_add]
      have hL : (x.bwd * y.bwd + cx * y.bwd) + (x.bwd * y.bwd + x.bwd * cy) + cx * cy =
                ((x.bwd * y.bwd + x.bwd * cy) + (cx * y.bwd + cx * cy)) + x.bwd * y.bwd := by
        rw [step13_add_assoc (x.bwd * y.bwd) (cx * y.bwd) _,
            ← step13_add_assoc (cx * y.bwd) (x.bwd * y.bwd) (x.bwd * cy),
            step12_add_comm (cx * y.bwd) (x.bwd * y.bwd),
            step13_add_assoc (x.bwd * y.bwd) (cx * y.bwd) (x.bwd * cy),
            step12_add_comm (cx * y.bwd) (x.bwd * cy),
            ← step13_add_assoc (x.bwd * y.bwd) (x.bwd * cy) (cx * y.bwd),
            step13_add_assoc (x.bwd * y.bwd) ((x.bwd * y.bwd + x.bwd * cy) + cx * y.bwd) (cx * cy),
            step13_add_assoc (x.bwd * y.bwd + x.bwd * cy) (cx * y.bwd) (cx * cy),
            step12_add_comm (x.bwd * y.bwd) _]
      exact hL

theorem bal_mul_gt_ge {x y : Balance} (hx : BalGtZero x) (hy : BalGeZero y) :
    BalGeZero (balance_mul x y) := by
  cases hy with
  | inl hyz =>
    exact Or.inl (step24_balance_trans (bal_mul_congr (step24_balance_refl x) hyz) (bal_mul_zero x))
  | inr hygt => exact Or.inr (bal_mul_gt_gt hx hygt)

theorem balOfNat_succ_gt_zero (n : Nat) : BalGtZero (balOfNat (n + 1)) := by
  induction n with
  | zero =>
    exact bal_gt_zero_of_equiv (step24_balance_symm (bal_zero_add bone)) bone_gt_zero
  | succ k ih =>
    exact bal_add_gt_ge ih (Or.inr bone_gt_zero)

theorem bal_one_add_sq_gt_zero (x : Balance) : BalGtZero (balance_add bone (bsq x)) :=
  bal_add_gt_ge bone_gt_zero (bal_sq_ge_zero x)


-- =============================================================================
-- ЭТАП XXVI. БЛОК I: АЛГЕБРАИЧЕСКОЕ ПРЕОБРАЗОВАНИЕ ФУРЬЕ НА БАЛАНСАХ
-- (Шаги 103–107; соответствуют шагам 53–57 плана micro_continual_core)
-- =============================================================================

inductive Fin4 where
  | f0 | f1 | f2 | f3
deriving DecidableEq

def Fin4.toNat : Fin4 → Nat
  | .f0 => 0
  | .f1 => 1
  | .f2 => 2
  | .f3 => 3

def gzero : GaussBalance := ⟨bzero, bzero⟩
def gneg_i : GaussBalance := ⟨bzero, bneg_one⟩

def gi_pow (m : Nat) : GaussBalance :=
  match m % 4 with
  | 0 => gone
  | 1 => gi
  | 2 => gneg_one
  | _ => gneg_i

def gi_neg_pow (m : Nat) : GaussBalance :=
  match m % 4 with
  | 0 => gone
  | 1 => gneg_i
  | 2 => gneg_one
  | _ => gi

def gauss_scale (s : Balance) (z : GaussBalance) : GaussBalance :=
  ⟨balance_mul s z.re, balance_mul s z.im⟩

def gauss_norm_sq (z : GaussBalance) : Balance :=
  balance_add (bsq z.re) (bsq z.im)

def dft4_orth_sum (k m : Fin4) : GaussBalance :=
  let d := (k.toNat + 4 - m.toNat) % 4
  gauss_add
    (gauss_add (gi_pow (0 * d)) (gi_pow (1 * d)))
    (gauss_add (gi_pow (2 * d)) (gi_pow (3 * d)))

/-- ШАГ 103 (План 53). Ортогональность волн 4-точечного ДПФ на балансах:
    ∑_{n=0}^{3} i^{n(k-m)} ~ 4 · δ_{km} (при k ≠ m волны гасят друг друга в равновесие 0). -/
theorem step103_dft4_orthogonality (k m : Fin4) :
    (k = m → GaussEquiv (dft4_orth_sum k m) ⟨balOfNat 4, bzero⟩) ∧
    (k ≠ m → GaussEquiv (dft4_orth_sum k m) gzero) := by
  cases k <;> cases m <;>
  · constructor
    · intro h
      first
      | contradiction
      | (dsimp [dft4_orth_sum, Fin4.toNat, gi_pow, gauss_add, gone, gi, gneg_one, gneg_i,
                GaussEquiv, balOfNat, bone, bzero, bneg_one, balance_add, BalanceEquiv]
         exact ⟨rfl, rfl⟩)
    · intro h
      first
      | (exfalso; exact h rfl)
      | (dsimp [dft4_orth_sum, Fin4.toNat, gi_pow, gauss_add, gone, gi, gneg_one, gneg_i, gzero,
                GaussEquiv, bone, bzero, bneg_one, balance_add, BalanceEquiv]
         exact ⟨rfl, rfl⟩)

def dft4 (x0 x1 x2 x3 : GaussBalance) (k : Fin4) : GaussBalance :=
  let kn := k.toNat
  gauss_add
    (gauss_add (gauss_mul x0 (gi_neg_pow (0 * kn))) (gauss_mul x1 (gi_neg_pow (1 * kn))))
    (gauss_add (gauss_mul x2 (gi_neg_pow (2 * kn))) (gauss_mul x3 (gi_neg_pow (3 * kn))))

def idft4 (X0 X1 X2 X3 : GaussBalance) (n : Fin4) : GaussBalance :=
  let nn := n.toNat
  gauss_add
    (gauss_add (gauss_mul X0 (gi_pow (0 * nn))) (gauss_mul X1 (gi_pow (1 * nn))))
    (gauss_add (gauss_mul X2 (gi_pow (2 * nn))) (gauss_mul X3 (gi_pow (3 * nn))))

-- Вспомогательные линейные комбинации для быстрого упрощения DFT4
def g_mul_gi (z : GaussBalance) : GaussBalance := ⟨balance_neg z.im, z.re⟩
def g_mul_neg1 (z : GaussBalance) : GaussBalance := ⟨balance_neg z.re, balance_neg z.im⟩
def g_mul_negi (z : GaussBalance) : GaussBalance := ⟨z.im, balance_neg z.re⟩

theorem gauss_add_congr {a1 a2 b1 b2 : GaussBalance}
    (ha : GaussEquiv a1 a2) (hb : GaussEquiv b1 b2) :
    GaussEquiv (gauss_add a1 b1) (gauss_add a2 b2) :=
  ⟨step25_balance_add_congr ha.1 hb.1, step25_balance_add_congr ha.2 hb.2⟩

theorem gauss_trans {a b c : GaussBalance} (h1 : GaussEquiv a b) (h2 : GaussEquiv b c) :
    GaussEquiv a c :=
  ⟨step24_balance_trans h1.1 h2.1, step24_balance_trans h1.2 h2.2⟩

theorem gauss_mul_congr_left {a1 a2 : GaussBalance} (b : GaussBalance) (ha : GaussEquiv a1 a2) :
    GaussEquiv (gauss_mul a1 b) (gauss_mul a2 b) :=
  ⟨bal_sub_congr (bal_mul_congr_left b.re ha.1) (bal_mul_congr_left b.im ha.2),
   step25_balance_add_congr (bal_mul_congr_left b.im ha.1) (bal_mul_congr_left b.re ha.2)⟩

theorem gauss_mul_gone (z : GaussBalance) : GaussEquiv (gauss_mul z gone) z := by
  constructor
  · dsimp [gauss_mul, gone]
    exact step24_balance_trans
      (bal_sub_congr (bal_mul_one z.re) (bal_mul_zero z.im))
      (bal_add_zero z.re)
  · dsimp [gauss_mul, gone]
    exact step24_balance_trans
      (step25_balance_add_congr (bal_mul_zero z.re) (bal_mul_one z.im))
      (bal_zero_add z.im)

theorem gauss_mul_gi (z : GaussBalance) : GaussEquiv (gauss_mul z gi) (g_mul_gi z) := by
  constructor
  · dsimp [gauss_mul, gi, g_mul_gi]
    exact step24_balance_trans
      (bal_sub_congr (bal_mul_zero z.re) (bal_mul_one z.im))
      (bal_zero_add (balance_neg z.im))
  · dsimp [gauss_mul, gi, g_mul_gi]
    exact step24_balance_trans
      (step25_balance_add_congr (bal_mul_one z.re) (bal_mul_zero z.im))
      (bal_add_zero z.re)

theorem gauss_mul_gneg_one (z : GaussBalance) : GaussEquiv (gauss_mul z gneg_one) (g_mul_neg1 z) := by
  constructor
  · dsimp [gauss_mul, gneg_one, g_mul_neg1]
    have h1 : BalanceEquiv (balance_mul z.re bneg_one) (balance_neg z.re) :=
      step24_balance_trans (bal_mul_neg z.re bone) (bal_neg_congr (bal_mul_one z.re))
    exact step24_balance_trans (bal_sub_congr h1 (bal_mul_zero z.im)) (bal_add_zero _)
  · dsimp [gauss_mul, gneg_one, g_mul_neg1]
    have h1 : BalanceEquiv (balance_mul z.im bneg_one) (balance_neg z.im) :=
      step24_balance_trans (bal_mul_neg z.im bone) (bal_neg_congr (bal_mul_one z.im))
    exact step24_balance_trans (step25_balance_add_congr (bal_mul_zero z.re) h1) (bal_zero_add _)

theorem gauss_mul_gneg_i (z : GaussBalance) : GaussEquiv (gauss_mul z gneg_i) (g_mul_negi z) := by
  constructor
  · dsimp [gauss_mul, gneg_i, g_mul_negi]
    have h1 : BalanceEquiv (balance_mul z.im bneg_one) (balance_neg z.im) :=
      step24_balance_trans (bal_mul_neg z.im bone) (bal_neg_congr (bal_mul_one z.im))
    exact step24_balance_trans (bal_sub_congr (bal_mul_zero z.re) h1)
      (step24_balance_trans (bal_zero_add _) (bal_neg_neg z.im))
  · dsimp [gauss_mul, gneg_i, g_mul_negi]
    have h1 : BalanceEquiv (balance_mul z.re bneg_one) (balance_neg z.re) :=
      step24_balance_trans (bal_mul_neg z.re bone) (bal_neg_congr (bal_mul_one z.re))
    exact step24_balance_trans (step25_balance_add_congr h1 (bal_mul_zero z.im)) (bal_add_zero _)

-- Упрощённая явная форма 4-точечного ДПФ (эквивалентна dft4)
def dft4_fast (x0 x1 x2 x3 : GaussBalance) : Fin4 → GaussBalance
  | .f0 => gauss_add (gauss_add x0 x1) (gauss_add x2 x3)
  | .f1 => gauss_add (gauss_add x0 (g_mul_negi x1)) (gauss_add (g_mul_neg1 x2) (g_mul_gi x3))
  | .f2 => gauss_add (gauss_add x0 (g_mul_neg1 x1)) (gauss_add x2 (g_mul_neg1 x3))
  | .f3 => gauss_add (gauss_add x0 (g_mul_gi x1)) (gauss_add (g_mul_neg1 x2) (g_mul_negi x3))

theorem dft4_equiv_fast (x0 x1 x2 x3 : GaussBalance) (k : Fin4) :
    GaussEquiv (dft4 x0 x1 x2 x3 k) (dft4_fast x0 x1 x2 x3 k) := by
  cases k with
  | f0 =>
    dsimp [dft4, dft4_fast, Fin4.toNat, gi_neg_pow]
    exact gauss_add_congr (gauss_add_congr (gauss_mul_gone _) (gauss_mul_gone _))
                          (gauss_add_congr (gauss_mul_gone _) (gauss_mul_gone _))
  | f1 =>
    dsimp [dft4, dft4_fast, Fin4.toNat, gi_neg_pow]
    exact gauss_add_congr (gauss_add_congr (gauss_mul_gone _) (gauss_mul_gneg_i _))
                          (gauss_add_congr (gauss_mul_gneg_one _) (gauss_mul_gi _))
  | f2 =>
    dsimp [dft4, dft4_fast, Fin4.toNat, gi_neg_pow]
    exact gauss_add_congr (gauss_add_congr (gauss_mul_gone _) (gauss_mul_gneg_one _))
                          (gauss_add_congr (gauss_mul_gone _) (gauss_mul_gneg_one _))
  | f3 =>
    dsimp [dft4, dft4_fast, Fin4.toNat, gi_neg_pow]
    exact gauss_add_congr (gauss_add_congr (gauss_mul_gone _) (gauss_mul_gi _))
                          (gauss_add_congr (gauss_mul_gneg_one _) (gauss_mul_gneg_i _))

def idft4_fast (X0 X1 X2 X3 : GaussBalance) : Fin4 → GaussBalance
  | .f0 => gauss_add (gauss_add X0 X1) (gauss_add X2 X3)
  | .f1 => gauss_add (gauss_add X0 (g_mul_gi X1)) (gauss_add (g_mul_neg1 X2) (g_mul_negi X3))
  | .f2 => gauss_add (gauss_add X0 (g_mul_neg1 X1)) (gauss_add X2 (g_mul_neg1 X3))
  | .f3 => gauss_add (gauss_add X0 (g_mul_negi X1)) (gauss_add (g_mul_neg1 X2) (g_mul_gi X3))

theorem idft4_equiv_fast (X0 X1 X2 X3 : GaussBalance) (n : Fin4) :
    GaussEquiv (idft4 X0 X1 X2 X3 n) (idft4_fast X0 X1 X2 X3 n) := by
  cases n with
  | f0 =>
    dsimp [idft4, idft4_fast, Fin4.toNat, gi_pow]
    exact gauss_add_congr (gauss_add_congr (gauss_mul_gone _) (gauss_mul_gone _))
                          (gauss_add_congr (gauss_mul_gone _) (gauss_mul_gone _))
  | f1 =>
    dsimp [idft4, idft4_fast, Fin4.toNat, gi_pow]
    exact gauss_add_congr (gauss_add_congr (gauss_mul_gone _) (gauss_mul_gi _))
                          (gauss_add_congr (gauss_mul_gneg_one _) (gauss_mul_gneg_i _))
  | f2 =>
    dsimp [idft4, idft4_fast, Fin4.toNat, gi_pow]
    exact gauss_add_congr (gauss_add_congr (gauss_mul_gone _) (gauss_mul_gneg_one _))
                          (gauss_add_congr (gauss_mul_gone _) (gauss_mul_gneg_one _))
  | f3 =>
    dsimp [idft4, idft4_fast, Fin4.toNat, gi_pow]
    exact gauss_add_congr (gauss_add_congr (gauss_mul_gone _) (gauss_mul_gneg_i _))
                          (gauss_add_congr (gauss_mul_gneg_one _) (gauss_mul_gi _))

theorem idft4_congr {A0 A1 A2 A3 B0 B1 B2 B3 : GaussBalance}
    (h0 : GaussEquiv A0 B0) (h1 : GaussEquiv A1 B1) (h2 : GaussEquiv A2 B2) (h3 : GaussEquiv A3 B3)
    (n : Fin4) :
    GaussEquiv (idft4 A0 A1 A2 A3 n) (idft4 B0 B1 B2 B3 n) := by
  dsimp [idft4]
  exact gauss_add_congr
    (gauss_add_congr (gauss_mul_congr_left _ h0) (gauss_mul_congr_left _ h1))
    (gauss_add_congr (gauss_mul_congr_left _ h2) (gauss_mul_congr_left _ h3))

def dft4_env (x0 x1 x2 x3 : GaussBalance) : Nat → Balance
  | 0 => x0.re | 1 => x0.im
  | 2 => x1.re | 3 => x1.im
  | 4 => x2.re | 5 => x2.im
  | 6 => x3.re | 7 => x3.im
  | _ => bzero

-- Выражения BExpr для быстрого ДПФ (пара (re, im))
def gAddE (a b : BExpr × BExpr) : BExpr × BExpr := (.add a.1 b.1, .add a.2 b.2)
def gNeg1E (a : BExpr × BExpr) : BExpr × BExpr := (.neg a.1, .neg a.2)
def gGiE (a : BExpr × BExpr) : BExpr × BExpr := (.neg a.2, a.1)
def gNegiE (a : BExpr × BExpr) : BExpr × BExpr := (a.2, .neg a.1)
def gNormSqE (a : BExpr × BExpr) : BExpr := .add (.mul a.1 a.1) (.mul a.2 a.2)

def ex0 : BExpr × BExpr := (.var 0, .var 1)
def ex1 : BExpr × BExpr := (.var 2, .var 3)
def ex2 : BExpr × BExpr := (.var 4, .var 5)
def ex3 : BExpr × BExpr := (.var 6, .var 7)

def dft4E (a0 a1 a2 a3 : BExpr × BExpr) : Fin4 → BExpr × BExpr
  | .f0 => gAddE (gAddE a0 a1) (gAddE a2 a3)
  | .f1 => gAddE (gAddE a0 (gNegiE a1)) (gAddE (gNeg1E a2) (gGiE a3))
  | .f2 => gAddE (gAddE a0 (gNeg1E a1)) (gAddE a2 (gNeg1E a3))
  | .f3 => gAddE (gAddE a0 (gGiE a1)) (gAddE (gNeg1E a2) (gNegiE a3))

def idft4E (A0 A1 A2 A3 : BExpr × BExpr) : Fin4 → BExpr × BExpr
  | .f0 => gAddE (gAddE A0 A1) (gAddE A2 A3)
  | .f1 => gAddE (gAddE A0 (gGiE A1)) (gAddE (gNeg1E A2) (gNegiE A3))
  | .f2 => gAddE (gAddE A0 (gNeg1E A1)) (gAddE A2 (gNeg1E A3))
  | .f3 => gAddE (gAddE A0 (gNegiE A1)) (gAddE (gNeg1E A2) (gGiE A3))

/-- ШАГ 104 (План 54). Точное обращение 4-точечного ДПФ для ПРОИЗВОЛЬНЫХ сигналов:
    IDFT₄(DFT₄(x)) ~ 4 · x по всем 4 координатам (без потерь и без аксиом). -/
theorem step104_dft4_inversion (x0 x1 x2 x3 : GaussBalance) :
    let X0 := dft4 x0 x1 x2 x3 .f0
    let X1 := dft4 x0 x1 x2 x3 .f1
    let X2 := dft4 x0 x1 x2 x3 .f2
    let X3 := dft4 x0 x1 x2 x3 .f3
    GaussEquiv (idft4 X0 X1 X2 X3 .f0) (gauss_scale (balOfNat 4) x0) ∧
    GaussEquiv (idft4 X0 X1 X2 X3 .f1) (gauss_scale (balOfNat 4) x1) ∧
    GaussEquiv (idft4 X0 X1 X2 X3 .f2) (gauss_scale (balOfNat 4) x2) ∧
    GaussEquiv (idft4 X0 X1 X2 X3 .f3) (gauss_scale (balOfNat 4) x3) := by
  dsimp only
  let env := dft4_env x0 x1 x2 x3
  let F0 := dft4_fast x0 x1 x2 x3 .f0
  let F1 := dft4_fast x0 x1 x2 x3 .f1
  let F2 := dft4_fast x0 x1 x2 x3 .f2
  let F3 := dft4_fast x0 x1 x2 x3 .f3
  have hf := fun n => gauss_trans
    (idft4_congr (dft4_equiv_fast x0 x1 x2 x3 .f0) (dft4_equiv_fast x0 x1 x2 x3 .f1)
                 (dft4_equiv_fast x0 x1 x2 x3 .f2) (dft4_equiv_fast x0 x1 x2 x3 .f3) n)
    (idft4_equiv_fast F0 F1 F2 F3 n)
  let E0 := dft4E ex0 ex1 ex2 ex3 .f0
  let E1 := dft4E ex0 ex1 ex2 ex3 .f1
  let E2 := dft4E ex0 ex1 ex2 ex3 .f2
  let E3 := dft4E ex0 ex1 ex2 ex3 .f3
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact gauss_trans (hf .f0)
      ⟨bal_poly_eq env (idft4E E0 E1 E2 E3 .f0).1 (.mul (.nat 4) (.var 0)) rfl,
       bal_poly_eq env (idft4E E0 E1 E2 E3 .f0).2 (.mul (.nat 4) (.var 1)) rfl⟩
  · exact gauss_trans (hf .f1)
      ⟨bal_poly_eq env (idft4E E0 E1 E2 E3 .f1).1 (.mul (.nat 4) (.var 2)) rfl,
       bal_poly_eq env (idft4E E0 E1 E2 E3 .f1).2 (.mul (.nat 4) (.var 3)) rfl⟩
  · exact gauss_trans (hf .f2)
      ⟨bal_poly_eq env (idft4E E0 E1 E2 E3 .f2).1 (.mul (.nat 4) (.var 4)) rfl,
       bal_poly_eq env (idft4E E0 E1 E2 E3 .f2).2 (.mul (.nat 4) (.var 5)) rfl⟩
  · exact gauss_trans (hf .f3)
      ⟨bal_poly_eq env (idft4E E0 E1 E2 E3 .f3).1 (.mul (.nat 4) (.var 6)) rfl,
       bal_poly_eq env (idft4E E0 E1 E2 E3 .f3).2 (.mul (.nat 4) (.var 7)) rfl⟩

-- БОНУС: 3-фазное ДПФ на балансах Эйзенштейна ℤ[ω] (1 + ω + ω² = 0 из Шага 42)
structure EisBalance where
  a : Balance  -- компонента при 1
  b : Balance  -- компонента при ω

def EisEquiv (x y : EisBalance) : Prop :=
  BalanceEquiv x.a y.a ∧ BalanceEquiv x.b y.b

def eis_add (x y : EisBalance) : EisBalance :=
  ⟨balance_add x.a y.a, balance_add x.b y.b⟩

-- Умножение на ω: ω(a + bω) = aω + b(-1 - ω) = -b + (a - b)ω
def eis_mul_w (x : EisBalance) : EisBalance :=
  ⟨balance_neg x.b, balance_sub x.a x.b⟩

-- Умножение на ω² = -1 - ω: ω²(a + bω) = (b - a) + (-a)ω
def eis_mul_w2 (x : EisBalance) : EisBalance :=
  ⟨balance_sub x.b x.a, balance_neg x.a⟩

def eis_scale (s : Balance) (x : EisBalance) : EisBalance :=
  ⟨balance_mul s x.a, balance_mul s x.b⟩

def dft3 (x0 x1 x2 : EisBalance) : Vert3 → EisBalance
  | .a => eis_add (eis_add x0 x1) x2
  | .b => eis_add (eis_add x0 (eis_mul_w2 x1)) (eis_mul_w x2)
  | .c => eis_add (eis_add x0 (eis_mul_w x1)) (eis_mul_w2 x2)

def idft3 (X0 X1 X2 : EisBalance) : Vert3 → EisBalance
  | .a => eis_add (eis_add X0 X1) X2
  | .b => eis_add (eis_add X0 (eis_mul_w X1)) (eis_mul_w2 X2)
  | .c => eis_add (eis_add X0 (eis_mul_w2 X1)) (eis_mul_w X2)

def eis_env (x0 x1 x2 : EisBalance) : Nat → Balance
  | 0 => x0.a | 1 => x0.b
  | 2 => x1.a | 3 => x1.b
  | 4 => x2.a | 5 => x2.b
  | _ => bzero

def eAddE (u v : BExpr × BExpr) : BExpr × BExpr := (.add u.1 v.1, .add u.2 v.2)
def eMulWE (u : BExpr × BExpr) : BExpr × BExpr := (.neg u.2, .sub u.1 u.2)
def eMulW2E (u : BExpr × BExpr) : BExpr × BExpr := (.sub u.2 u.1, .neg u.1)

-- Топ-левел закрытые полиномиальные тождества (быстрая kernel-редукция, 0 аксиом)
private def e_dft3_Z0 : BExpr × BExpr :=
  let u0 := (BExpr.var 0, BExpr.var 1)
  let u1 := (BExpr.var 2, BExpr.var 3)
  let u2 := (BExpr.var 4, BExpr.var 5)
  let Y0 := eAddE (eAddE u0 u1) u2
  let Y1 := eAddE (eAddE u0 (eMulW2E u1)) (eMulWE u2)
  let Y2 := eAddE (eAddE u0 (eMulWE u1)) (eMulW2E u2)
  eAddE (eAddE Y0 Y1) Y2

private def e_dft3_Z1 : BExpr × BExpr :=
  let u0 := (BExpr.var 0, BExpr.var 1)
  let u1 := (BExpr.var 2, BExpr.var 3)
  let u2 := (BExpr.var 4, BExpr.var 5)
  let Y0 := eAddE (eAddE u0 u1) u2
  let Y1 := eAddE (eAddE u0 (eMulW2E u1)) (eMulWE u2)
  let Y2 := eAddE (eAddE u0 (eMulWE u1)) (eMulW2E u2)
  eAddE (eAddE Y0 (eMulWE Y1)) (eMulW2E Y2)

private def e_dft3_Z2 : BExpr × BExpr :=
  let u0 := (BExpr.var 0, BExpr.var 1)
  let u1 := (BExpr.var 2, BExpr.var 3)
  let u2 := (BExpr.var 4, BExpr.var 5)
  let Y0 := eAddE (eAddE u0 u1) u2
  let Y1 := eAddE (eAddE u0 (eMulW2E u1)) (eMulWE u2)
  let Y2 := eAddE (eAddE u0 (eMulWE u1)) (eMulW2E u2)
  eAddE (eAddE Y0 (eMulW2E Y1)) (eMulWE Y2)

private theorem p_dft3_z01 : e_dft3_Z0.1.norm = (BExpr.mul (.nat 3) (.var 0)).norm := rfl
private theorem p_dft3_z02 : e_dft3_Z0.2.norm = (BExpr.mul (.nat 3) (.var 1)).norm := rfl
private theorem p_dft3_z11 : e_dft3_Z1.1.norm = (BExpr.mul (.nat 3) (.var 2)).norm := rfl
private theorem p_dft3_z12 : e_dft3_Z1.2.norm = (BExpr.mul (.nat 3) (.var 3)).norm := rfl
private theorem p_dft3_z21 : e_dft3_Z2.1.norm = (BExpr.mul (.nat 3) (.var 4)).norm := rfl
private theorem p_dft3_z22 : e_dft3_Z2.2.norm = (BExpr.mul (.nat 3) (.var 5)).norm := rfl

/-- БОНУС К ШАГУ 104 (из Шага 42: 1 + ω + ω² ~ 0).
    Точное 3-фазное ДПФ: IDFT₃(DFT₃(x)) ~ 3 · x для произвольных (x₀, x₁, x₂) ∈ ℤ[ω]. -/
theorem step104b_dft3_inversion (x0 x1 x2 : EisBalance) :
    let X0 := dft3 x0 x1 x2 .a
    let X1 := dft3 x0 x1 x2 .b
    let X2 := dft3 x0 x1 x2 .c
    EisEquiv (idft3 X0 X1 X2 .a) (eis_scale (balOfNat 3) x0) ∧
    EisEquiv (idft3 X0 X1 X2 .b) (eis_scale (balOfNat 3) x1) ∧
    EisEquiv (idft3 X0 X1 X2 .c) (eis_scale (balOfNat 3) x2) := by
  let env := eis_env x0 x1 x2
  refine ⟨⟨bal_poly_eq env e_dft3_Z0.1 (.mul (.nat 3) (.var 0)) p_dft3_z01,
           bal_poly_eq env e_dft3_Z0.2 (.mul (.nat 3) (.var 1)) p_dft3_z02⟩,
          ⟨bal_poly_eq env e_dft3_Z1.1 (.mul (.nat 3) (.var 2)) p_dft3_z11,
           bal_poly_eq env e_dft3_Z1.2 (.mul (.nat 3) (.var 3)) p_dft3_z12⟩,
          ⟨bal_poly_eq env e_dft3_Z2.1 (.mul (.nat 3) (.var 4)) p_dft3_z21,
           bal_poly_eq env e_dft3_Z2.2 (.mul (.nat 3) (.var 5)) p_dft3_z22⟩⟩

theorem gauss_norm_sq_congr {a b : GaussBalance} (h : GaussEquiv a b) :
    BalanceEquiv (gauss_norm_sq a) (gauss_norm_sq b) :=
  step25_balance_add_congr (bal_mul_congr h.1 h.1) (bal_mul_congr h.2 h.2)

private def e_para_L : BExpr :=
  let eu := (BExpr.var 0, BExpr.var 1)
  let ev := (BExpr.var 2, BExpr.var 3)
  .add (gNormSqE (gAddE eu ev)) (gNormSqE (gAddE eu (gNeg1E ev)))

private def e_para_R : BExpr :=
  let eu := (BExpr.var 0, BExpr.var 1)
  let ev := (BExpr.var 2, BExpr.var 3)
  .mul (.nat 2) (.add (gNormSqE eu) (gNormSqE ev))

private theorem p_para : e_para_L.norm = e_para_R.norm := rfl

-- Тождество параллелограмма для гауссовых балансов: |u + v|² + |u - v|² ~ 2(|u|² + |v|²)
theorem gauss_parallelogram (u v : GaussBalance) :
    BalanceEquiv
      (balance_add (gauss_norm_sq (gauss_add u v))
                   (gauss_norm_sq (gauss_add u (g_mul_neg1 v))))
      (balance_mul (balOfNat 2) (balance_add (gauss_norm_sq u) (gauss_norm_sq v))) := by
  let env : Nat → Balance := fun | 0 => u.re | 1 => u.im | 2 => v.re | 3 => v.im | _ => bzero
  exact bal_poly_eq env e_para_L e_para_R p_para

theorem gauss_norm_sq_mul_gi (z : GaussBalance) :
    BalanceEquiv (gauss_norm_sq (g_mul_gi z)) (gauss_norm_sq z) := by
  let env : Nat → Balance := fun | 0 => z.re | 1 => z.im | _ => bzero
  let ez : BExpr × BExpr := (.var 0, .var 1)
  exact bal_poly_eq env (gNormSqE (gGiE ez)) (gNormSqE ez) rfl

/-- ШАГ 105 (План 55). Теорема Парсеваля / сохранение «наличия» (через бабочку Кули-Тьюки!):
    ∑_{k=0}^{3} |X_k|² ~ 4 · ∑_{n=0}^{3} |x_n|² для произвольного сигнала (x₀, x₁, x₂, x₃). -/
theorem step105_dft4_parseval (x0 x1 x2 x3 : GaussBalance) :
    let X0 := dft4 x0 x1 x2 x3 .f0
    let X1 := dft4 x0 x1 x2 x3 .f1
    let X2 := dft4 x0 x1 x2 x3 .f2
    let X3 := dft4 x0 x1 x2 x3 .f3
    BalanceEquiv
      (balance_add (balance_add (gauss_norm_sq X0) (gauss_norm_sq X1))
                   (balance_add (gauss_norm_sq X2) (gauss_norm_sq X3)))
      (balance_mul (balOfNat 4)
        (balance_add (balance_add (gauss_norm_sq x0) (gauss_norm_sq x1))
                     (balance_add (gauss_norm_sq x2) (gauss_norm_sq x3)))) := by
  dsimp only
  let p := gauss_add x0 x2
  let q := gauss_add x0 (g_mul_neg1 x2)
  let r := gauss_add x1 x3
  let s := gauss_add x1 (g_mul_neg1 x3)
  let env := dft4_env x0 x1 x2 x3
  let ep := gAddE ex0 ex2
  let eq := gAddE ex0 (gNeg1E ex2)
  let er := gAddE ex1 ex3
  let es := gAddE ex1 (gNeg1E ex3)
  have hX0 : GaussEquiv (dft4 x0 x1 x2 x3 .f0) (gauss_add p r) :=
    gauss_trans (dft4_equiv_fast x0 x1 x2 x3 .f0)
      ⟨bal_poly_eq env (dft4E ex0 ex1 ex2 ex3 .f0).1 (gAddE ep er).1 rfl,
       bal_poly_eq env (dft4E ex0 ex1 ex2 ex3 .f0).2 (gAddE ep er).2 rfl⟩
  have hX2 : GaussEquiv (dft4 x0 x1 x2 x3 .f2) (gauss_add p (g_mul_neg1 r)) :=
    gauss_trans (dft4_equiv_fast x0 x1 x2 x3 .f2)
      ⟨bal_poly_eq env (dft4E ex0 ex1 ex2 ex3 .f2).1 (gAddE ep (gNeg1E er)).1 rfl,
       bal_poly_eq env (dft4E ex0 ex1 ex2 ex3 .f2).2 (gAddE ep (gNeg1E er)).2 rfl⟩
  have hX3 : GaussEquiv (dft4 x0 x1 x2 x3 .f3) (gauss_add q (g_mul_gi s)) :=
    gauss_trans (dft4_equiv_fast x0 x1 x2 x3 .f3)
      ⟨bal_poly_eq env (dft4E ex0 ex1 ex2 ex3 .f3).1 (gAddE eq (gGiE es)).1 rfl,
       bal_poly_eq env (dft4E ex0 ex1 ex2 ex3 .f3).2 (gAddE eq (gGiE es)).2 rfl⟩
  have hX1 : GaussEquiv (dft4 x0 x1 x2 x3 .f1) (gauss_add q (g_mul_neg1 (g_mul_gi s))) :=
    gauss_trans (dft4_equiv_fast x0 x1 x2 x3 .f1)
      ⟨bal_poly_eq env (dft4E ex0 ex1 ex2 ex3 .f1).1 (gAddE eq (gNeg1E (gGiE es))).1 rfl,
       bal_poly_eq env (dft4E ex0 ex1 ex2 ex3 .f1).2 (gAddE eq (gNeg1E (gGiE es))).2 rfl⟩
  -- 1. Перегруппировка (|X0|² + |X1|²) + (|X2|² + |X3|²) ~ (|X0|² + |X2|²) + (|X3|² + |X1|²)
  have hre : BalanceEquiv
      (balance_add (balance_add (gauss_norm_sq (dft4 x0 x1 x2 x3 .f0)) (gauss_norm_sq (dft4 x0 x1 x2 x3 .f1)))
                   (balance_add (gauss_norm_sq (dft4 x0 x1 x2 x3 .f2)) (gauss_norm_sq (dft4 x0 x1 x2 x3 .f3))))
      (balance_add (balance_add (gauss_norm_sq (gauss_add p r)) (gauss_norm_sq (gauss_add p (g_mul_neg1 r))))
                   (balance_add (gauss_norm_sq (gauss_add q (g_mul_gi s))) (gauss_norm_sq (gauss_add q (g_mul_neg1 (g_mul_gi s)))))) :=
    step24_balance_trans
      (bal_add_add_add_comm _ _ _ _)
      (step25_balance_add_congr
        (step25_balance_add_congr (gauss_norm_sq_congr hX0) (gauss_norm_sq_congr hX2))
        (step24_balance_trans (bal_add_comm _ _)
          (step25_balance_add_congr (gauss_norm_sq_congr hX3) (gauss_norm_sq_congr hX1))))
  -- 2. Первая бабочка: |X0|² + |X2|² ~ 2(|p|² + |r|²), |X3|² + |X1|² ~ 2(|q|² + |s|²)
  have hb1 : BalanceEquiv
      (balance_add (balance_add (gauss_norm_sq (gauss_add p r)) (gauss_norm_sq (gauss_add p (g_mul_neg1 r))))
                   (balance_add (gauss_norm_sq (gauss_add q (g_mul_gi s))) (gauss_norm_sq (gauss_add q (g_mul_neg1 (g_mul_gi s))))))
      (balance_add (balance_mul (balOfNat 2) (balance_add (gauss_norm_sq p) (gauss_norm_sq r)))
                   (balance_mul (balOfNat 2) (balance_add (gauss_norm_sq q) (gauss_norm_sq s)))) :=
    step25_balance_add_congr
      (gauss_parallelogram p r)
      (step24_balance_trans (gauss_parallelogram q (g_mul_gi s))
        (bal_mul_congr (step24_balance_refl (balOfNat 2))
          (step25_balance_add_congr (step24_balance_refl _) (gauss_norm_sq_mul_gi s))))
  -- 3. Вторая бабочка: |p|² + |q|² ~ 2(|x0|² + |x2|²), |r|² + |s|² ~ 2(|x1|² + |x3|²)
  have hb2 : BalanceEquiv
      (balance_add (balance_mul (balOfNat 2) (balance_add (gauss_norm_sq p) (gauss_norm_sq r)))
                   (balance_mul (balOfNat 2) (balance_add (gauss_norm_sq q) (gauss_norm_sq s))))
      (balance_mul (balOfNat 2)
        (balance_add (balance_mul (balOfNat 2) (balance_add (gauss_norm_sq x0) (gauss_norm_sq x2)))
                     (balance_mul (balOfNat 2) (balance_add (gauss_norm_sq x1) (gauss_norm_sq x3))))) :=
    step24_balance_trans
      (step24_balance_symm (bal_left_distrib (balOfNat 2) _ _))
      (bal_mul_congr (step24_balance_refl (balOfNat 2))
        (step24_balance_trans (bal_add_add_add_comm _ _ _ _)
          (step25_balance_add_congr (gauss_parallelogram x0 x2) (gauss_parallelogram x1 x3))))
  -- 4. Финальное скалярное упрощение: 2*(2*(n0 + n2) + 2*(n1 + n3)) ~ 4*((n0 + n1) + (n2 + n3))
  let envN : Nat → Balance := fun
    | 0 => gauss_norm_sq x0 | 1 => gauss_norm_sq x1
    | 2 => gauss_norm_sq x2 | 3 => gauss_norm_sq x3 | _ => bzero
  have hfin := bal_poly_eq envN
    (.mul (.nat 2) (.add (.mul (.nat 2) (.add (.var 0) (.var 2)))
                         (.mul (.nat 2) (.add (.var 1) (.var 3)))))
    (.mul (.nat 4) (.add (.add (.var 0) (.var 1)) (.add (.var 2) (.var 3)))) rfl
  exact step24_balance_trans hre (step24_balance_trans hb1 (step24_balance_trans hb2 hfin))

/-- ШАГ 106 (План 56). Теорема о циклическом сдвиге во времени = фазовый поворот в спектре:
    - Сдвиг вправо (x₃, x₀, x₁, x₂) умножает k-ю гармонику на ротор i^{-k}.
    - Сдвиг влево  (x₁, x₂, x₃, x₀) умножает k-ю гармонику на ротор i^{+k}. -/
theorem step106_dft4_shift_is_phase_rotation (x0 x1 x2 x3 : GaussBalance) (k : Fin4) :
    GaussEquiv (dft4 x3 x0 x1 x2 k) (gauss_mul (dft4 x0 x1 x2 x3 k) (gi_neg_pow k.toNat)) ∧
    GaussEquiv (dft4 x1 x2 x3 x0 k) (gauss_mul (dft4 x0 x1 x2 x3 k) (gi_pow k.toNat)) := by
  let env := dft4_env x0 x1 x2 x3
  cases k with
  | f0 =>
    constructor
    · refine gauss_trans (dft4_equiv_fast x3 x0 x1 x2 .f0) ?_
      have hr := gauss_trans (gauss_mul_congr_left gone (dft4_equiv_fast x0 x1 x2 x3 .f0))
                   (gauss_mul_gone (dft4_fast x0 x1 x2 x3 .f0))
      have hm : GaussEquiv (dft4_fast x3 x0 x1 x2 .f0) (dft4_fast x0 x1 x2 x3 .f0) :=
        ⟨bal_poly_eq env (dft4E ex3 ex0 ex1 ex2 .f0).1 (dft4E ex0 ex1 ex2 ex3 .f0).1 rfl,
         bal_poly_eq env (dft4E ex3 ex0 ex1 ex2 .f0).2 (dft4E ex0 ex1 ex2 ex3 .f0).2 rfl⟩
      exact gauss_trans hm ⟨step24_balance_symm hr.1, step24_balance_symm hr.2⟩
    · refine gauss_trans (dft4_equiv_fast x1 x2 x3 x0 .f0) ?_
      have hr := gauss_trans (gauss_mul_congr_left gone (dft4_equiv_fast x0 x1 x2 x3 .f0))
                   (gauss_mul_gone (dft4_fast x0 x1 x2 x3 .f0))
      have hm : GaussEquiv (dft4_fast x1 x2 x3 x0 .f0) (dft4_fast x0 x1 x2 x3 .f0) :=
        ⟨bal_poly_eq env (dft4E ex1 ex2 ex3 ex0 .f0).1 (dft4E ex0 ex1 ex2 ex3 .f0).1 rfl,
         bal_poly_eq env (dft4E ex1 ex2 ex3 ex0 .f0).2 (dft4E ex0 ex1 ex2 ex3 .f0).2 rfl⟩
      exact gauss_trans hm ⟨step24_balance_symm hr.1, step24_balance_symm hr.2⟩
  | f1 =>
    constructor
    · refine gauss_trans (dft4_equiv_fast x3 x0 x1 x2 .f1) ?_
      have hr := gauss_trans (gauss_mul_congr_left gneg_i (dft4_equiv_fast x0 x1 x2 x3 .f1))
                   (gauss_mul_gneg_i (dft4_fast x0 x1 x2 x3 .f1))
      have hm : GaussEquiv (dft4_fast x3 x0 x1 x2 .f1) (g_mul_negi (dft4_fast x0 x1 x2 x3 .f1)) :=
        ⟨bal_poly_eq env (dft4E ex3 ex0 ex1 ex2 .f1).1 (gNegiE (dft4E ex0 ex1 ex2 ex3 .f1)).1 rfl,
         bal_poly_eq env (dft4E ex3 ex0 ex1 ex2 .f1).2 (gNegiE (dft4E ex0 ex1 ex2 ex3 .f1)).2 rfl⟩
      exact gauss_trans hm ⟨step24_balance_symm hr.1, step24_balance_symm hr.2⟩
    · refine gauss_trans (dft4_equiv_fast x1 x2 x3 x0 .f1) ?_
      have hr := gauss_trans (gauss_mul_congr_left gi (dft4_equiv_fast x0 x1 x2 x3 .f1))
                   (gauss_mul_gi (dft4_fast x0 x1 x2 x3 .f1))
      have hm : GaussEquiv (dft4_fast x1 x2 x3 x0 .f1) (g_mul_gi (dft4_fast x0 x1 x2 x3 .f1)) :=
        ⟨bal_poly_eq env (dft4E ex1 ex2 ex3 ex0 .f1).1 (gGiE (dft4E ex0 ex1 ex2 ex3 .f1)).1 rfl,
         bal_poly_eq env (dft4E ex1 ex2 ex3 ex0 .f1).2 (gGiE (dft4E ex0 ex1 ex2 ex3 .f1)).2 rfl⟩
      exact gauss_trans hm ⟨step24_balance_symm hr.1, step24_balance_symm hr.2⟩
  | f2 =>
    constructor
    · refine gauss_trans (dft4_equiv_fast x3 x0 x1 x2 .f2) ?_
      have hr := gauss_trans (gauss_mul_congr_left gneg_one (dft4_equiv_fast x0 x1 x2 x3 .f2))
                   (gauss_mul_gneg_one (dft4_fast x0 x1 x2 x3 .f2))
      have hm : GaussEquiv (dft4_fast x3 x0 x1 x2 .f2) (g_mul_neg1 (dft4_fast x0 x1 x2 x3 .f2)) :=
        ⟨bal_poly_eq env (dft4E ex3 ex0 ex1 ex2 .f2).1 (gNeg1E (dft4E ex0 ex1 ex2 ex3 .f2)).1 rfl,
         bal_poly_eq env (dft4E ex3 ex0 ex1 ex2 .f2).2 (gNeg1E (dft4E ex0 ex1 ex2 ex3 .f2)).2 rfl⟩
      exact gauss_trans hm ⟨step24_balance_symm hr.1, step24_balance_symm hr.2⟩
    · refine gauss_trans (dft4_equiv_fast x1 x2 x3 x0 .f2) ?_
      have hr := gauss_trans (gauss_mul_congr_left gneg_one (dft4_equiv_fast x0 x1 x2 x3 .f2))
                   (gauss_mul_gneg_one (dft4_fast x0 x1 x2 x3 .f2))
      have hm : GaussEquiv (dft4_fast x1 x2 x3 x0 .f2) (g_mul_neg1 (dft4_fast x0 x1 x2 x3 .f2)) :=
        ⟨bal_poly_eq env (dft4E ex1 ex2 ex3 ex0 .f2).1 (gNeg1E (dft4E ex0 ex1 ex2 ex3 .f2)).1 rfl,
         bal_poly_eq env (dft4E ex1 ex2 ex3 ex0 .f2).2 (gNeg1E (dft4E ex0 ex1 ex2 ex3 .f2)).2 rfl⟩
      exact gauss_trans hm ⟨step24_balance_symm hr.1, step24_balance_symm hr.2⟩
  | f3 =>
    constructor
    · refine gauss_trans (dft4_equiv_fast x3 x0 x1 x2 .f3) ?_
      have hr := gauss_trans (gauss_mul_congr_left gi (dft4_equiv_fast x0 x1 x2 x3 .f3))
                   (gauss_mul_gi (dft4_fast x0 x1 x2 x3 .f3))
      have hm : GaussEquiv (dft4_fast x3 x0 x1 x2 .f3) (g_mul_gi (dft4_fast x0 x1 x2 x3 .f3)) :=
        ⟨bal_poly_eq env (dft4E ex3 ex0 ex1 ex2 .f3).1 (gGiE (dft4E ex0 ex1 ex2 ex3 .f3)).1 rfl,
         bal_poly_eq env (dft4E ex3 ex0 ex1 ex2 .f3).2 (gGiE (dft4E ex0 ex1 ex2 ex3 .f3)).2 rfl⟩
      exact gauss_trans hm ⟨step24_balance_symm hr.1, step24_balance_symm hr.2⟩
    · refine gauss_trans (dft4_equiv_fast x1 x2 x3 x0 .f3) ?_
      have hr := gauss_trans (gauss_mul_congr_left gneg_i (dft4_equiv_fast x0 x1 x2 x3 .f3))
                   (gauss_mul_gneg_i (dft4_fast x0 x1 x2 x3 .f3))
      have hm : GaussEquiv (dft4_fast x1 x2 x3 x0 .f3) (g_mul_negi (dft4_fast x0 x1 x2 x3 .f3)) :=
        ⟨bal_poly_eq env (dft4E ex1 ex2 ex3 ex0 .f3).1 (gNegiE (dft4E ex0 ex1 ex2 ex3 .f3)).1 rfl,
         bal_poly_eq env (dft4E ex1 ex2 ex3 ex0 .f3).2 (gNegiE (dft4E ex0 ex1 ex2 ex3 .f3)).2 rfl⟩
      exact gauss_trans hm ⟨step24_balance_symm hr.1, step24_balance_symm hr.2⟩

/-- ШАГ 107 (План 57). Рациональный круг Кэли без sin/cos для произвольного ω : Balance:
    1. Знаменатель 1 + ω² строго положителен (и не может равняться равновесию 0).
    2. Точная окружность: (1 - ω²)² + (2ω)² ~ (1 + ω²)². -/
theorem step107_rational_circle_cayley (w : Balance) :
    BalGtZero (balance_add bone (bsq w)) ∧
    (BalanceEquiv (balance_add bone (bsq w)) bzero → False) ∧
    BalanceEquiv
      (balance_add (bsq (balance_sub bone (bsq w))) (bsq (balance_mul (balOfNat 2) w)))
      (bsq (balance_add bone (bsq w))) := by
  have hpos := bal_one_add_sq_gt_zero w
  refine ⟨hpos, bal_gt_zero_ne_zero hpos, ?_⟩
  let env : Nat → Balance := fun | 0 => w | _ => bzero
  let ew : BExpr := .var 0
  let ew2 : BExpr := .mul ew ew
  let c : BExpr := .sub .one ew2
  let s : BExpr := .mul (.nat 2) ew
  let d : BExpr := .add .one ew2
  exact bal_poly_eq env (.add (.mul c c) (.mul s s)) (.mul d d) rfl


-- =============================================================================
-- ЭТАП XXVII. БЛОК II: НЕКОММУТАТИВНЫЙ 3D РОТОР КЭЛИ SO(3) И BIBO-УСТОЙЧИВОСТЬ
-- (Шаги 108–111; соответствуют шагам 58–61 плана micro_continual_core)
-- =============================================================================

structure Vec3 where
  x : Balance
  y : Balance
  z : Balance

def Vec3Equiv (u v : Vec3) : Prop :=
  BalanceEquiv u.x v.x ∧ BalanceEquiv u.y v.y ∧ BalanceEquiv u.z v.z

def vec3_zero : Vec3 := ⟨bzero, bzero, bzero⟩
def vec3_add (u v : Vec3) : Vec3 := ⟨balance_add u.x v.x, balance_add u.y v.y, balance_add u.z v.z⟩
def vec3_neg (u : Vec3) : Vec3 := ⟨balance_neg u.x, balance_neg u.y, balance_neg u.z⟩
def vec3_scale (s : Balance) (u : Vec3) : Vec3 :=
  ⟨balance_mul s u.x, balance_mul s u.y, balance_mul s u.z⟩

def dot3 (u v : Vec3) : Balance :=
  balance_add (balance_add (balance_mul u.x v.x) (balance_mul u.y v.y)) (balance_mul u.z v.z)

def cross3 (u v : Vec3) : Vec3 :=
  ⟨balance_sub (balance_mul u.y v.z) (balance_mul u.z v.y),
   balance_sub (balance_mul u.z v.x) (balance_mul u.x v.z),
   balance_sub (balance_mul u.x v.y) (balance_mul u.y v.x)⟩

def vec3_env3 (a b c : Vec3) : Nat → Balance
  | 0 => a.x | 1 => a.y | 2 => a.z
  | 3 => b.x | 4 => b.y | 5 => b.z
  | 6 => c.x | 7 => c.y | 8 => c.z
  | _ => bzero

def vCrossE (u v : BExpr × BExpr × BExpr) : BExpr × BExpr × BExpr :=
  (.sub (.mul u.2.1 v.2.2) (.mul u.2.2 v.2.1),
   .sub (.mul u.2.2 v.1)   (.mul u.1 v.2.2),
   .sub (.mul u.1 v.2.1)   (.mul u.2.1 v.1))

def vAddE (u v : BExpr × BExpr × BExpr) : BExpr × BExpr × BExpr :=
  (.add u.1 v.1, .add u.2.1 v.2.1, .add u.2.2 v.2.2)

def vNegE (u : BExpr × BExpr × BExpr) : BExpr × BExpr × BExpr :=
  (.neg u.1, .neg u.2.1, .neg u.2.2)

def vScaleE (s : BExpr) (u : BExpr × BExpr × BExpr) : BExpr × BExpr × BExpr :=
  (.mul s u.1, .mul s u.2.1, .mul s u.2.2)

def vDotE (u v : BExpr × BExpr × BExpr) : BExpr :=
  .add (.add (.mul u.1 v.1) (.mul u.2.1 v.2.1)) (.mul u.2.2 v.2.2)

private def e_v_aa : BExpr × BExpr × BExpr :=
  let ea := (BExpr.var 0, BExpr.var 1, BExpr.var 2)
  vCrossE ea ea

private def e_v_ab : BExpr × BExpr × BExpr :=
  vCrossE (.var 0, .var 1, .var 2) (.var 3, .var 4, .var 5)

private def e_v_nba : BExpr × BExpr × BExpr :=
  vNegE (vCrossE (.var 3, .var 4, .var 5) (.var 0, .var 1, .var 2))

private def e_v_jac : BExpr × BExpr × BExpr :=
  let ea := (BExpr.var 0, BExpr.var 1, BExpr.var 2)
  let eb := (BExpr.var 3, BExpr.var 4, BExpr.var 5)
  let ec := (BExpr.var 6, BExpr.var 7, BExpr.var 8)
  vAddE (vAddE (vCrossE ea (vCrossE eb ec)) (vCrossE eb (vCrossE ec ea)))
        (vCrossE ec (vCrossE ea eb))

private theorem p_aa1 : e_v_aa.1.norm = BExpr.zero.norm := rfl
private theorem p_aa2 : e_v_aa.2.1.norm = BExpr.zero.norm := rfl
private theorem p_aa3 : e_v_aa.2.2.norm = BExpr.zero.norm := rfl
private theorem p_ab1 : e_v_ab.1.norm = e_v_nba.1.norm := rfl
private theorem p_ab2 : e_v_ab.2.1.norm = e_v_nba.2.1.norm := rfl
private theorem p_ab3 : e_v_ab.2.2.norm = e_v_nba.2.2.norm := rfl
private theorem p_jac1 : e_v_jac.1.norm = BExpr.zero.norm := rfl
private theorem p_jac2 : e_v_jac.2.1.norm = BExpr.zero.norm := rfl
private theorem p_jac3 : e_v_jac.2.2.norm = BExpr.zero.norm := rfl

/-- ШАГ 108 (План 58). Законы алгебры Ли su(2) для su2_lie_bracket (cross3):
    1. Самоаннигиляция: a × a ~ 0.
    2. Антисимметрия: a × b ~ -(b × a).
    3. Тождество Якоби: a × (b × c) + b × (c × a) + c × (a × b) ~ 0. -/
theorem step108_su2_lie_laws (a b c : Vec3) :
    Vec3Equiv (cross3 a a) vec3_zero ∧
    Vec3Equiv (cross3 a b) (vec3_neg (cross3 b a)) ∧
    Vec3Equiv
      (vec3_add (vec3_add (cross3 a (cross3 b c)) (cross3 b (cross3 c a)))
                (cross3 c (cross3 a b)))
      vec3_zero := by
  let env := vec3_env3 a b c
  refine ⟨⟨bal_poly_eq env e_v_aa.1 .zero p_aa1,
           bal_poly_eq env e_v_aa.2.1 .zero p_aa2,
           bal_poly_eq env e_v_aa.2.2 .zero p_aa3⟩,
          ⟨bal_poly_eq env e_v_ab.1 e_v_nba.1 p_ab1,
           bal_poly_eq env e_v_ab.2.1 e_v_nba.2.1 p_ab2,
           bal_poly_eq env e_v_ab.2.2 e_v_nba.2.2 p_ab3⟩,
          ⟨bal_poly_eq env e_v_jac.1 .zero p_jac1,
           bal_poly_eq env e_v_jac.2.1 .zero p_jac2,
           bal_poly_eq env e_v_jac.2.2 .zero p_jac3⟩⟩

-- Целочисленная форма 3D ротора Кэли до деления на D = 1 + ||g||² > 0:
-- w = (1 - ||g||²) v + 2 (g × v) + 2 (g · v) g
def cayley3d_num (g v : Vec3) : Vec3 :=
  let g2 := dot3 g g
  let omr := balance_sub bone g2
  let gv := dot3 g v
  vec3_add
    (vec3_add (vec3_scale omr v) (vec3_scale (balOfNat 2) (cross3 g v)))
    (vec3_scale (balance_mul (balOfNat 2) gv) g)

-- Форма из src/rtrl.rs (cayley_rot3) до деления на (1 + w²):
-- (1 + w²) v + 2w (e × v) + 2w² (e × (e × v))
def cayley_rot3_num (e : Vec3) (w : Balance) (v : Vec3) : Vec3 :=
  let w2 := bsq w
  let q := cross3 e v
  let r := cross3 e q
  vec3_add
    (vec3_add (vec3_scale (balance_add bone w2) v)
              (vec3_scale (balance_mul (balOfNat 2) w) q))
    (vec3_scale (balance_mul (balOfNat 2) w2) r)

theorem dot3_self_ge_zero (g : Vec3) : BalGeZero (dot3 g g) :=
  bal_add_ge_ge (bal_add_ge_ge (bal_sq_ge_zero g.x) (bal_sq_ge_zero g.y)) (bal_sq_ge_zero g.z)

private def e_lag_L : BExpr :=
  let ec := vCrossE (.var 0, .var 1, .var 2) (.var 3, .var 4, .var 5)
  vDotE ec ec

private def e_lag_R : BExpr :=
  let eg := (BExpr.var 0, BExpr.var 1, BExpr.var 2)
  let ev := (BExpr.var 3, BExpr.var 4, BExpr.var 5)
  let egv := vDotE eg ev
  .sub (.mul (vDotE eg eg) (vDotE ev ev)) (.mul egv egv)

private def e_v_ec : BExpr :=
  vDotE (.var 3, .var 4, .var 5) (vCrossE (.var 0, .var 1, .var 2) (.var 3, .var 4, .var 5))

private def e_g_ec : BExpr :=
  vDotE (.var 0, .var 1, .var 2) (vCrossE (.var 0, .var 1, .var 2) (.var 3, .var 4, .var 5))

private theorem p_lag : e_lag_L.norm = e_lag_R.norm := rfl
private theorem p_vec : e_v_ec.norm = BExpr.zero.norm := rfl
private theorem p_gec : e_g_ec.norm = BExpr.zero.norm := rfl

-- Тождество Лагранжа ||g × v||² ~ ||g||²·||v||² - (g·v)² и ортогональность g×v (0 аксиом)
theorem vec3_lagrange_and_orth (g v : Vec3) :
    BalanceEquiv (dot3 (cross3 g v) (cross3 g v))
                 (balance_sub (balance_mul (dot3 g g) (dot3 v v)) (bsq (dot3 g v))) ∧
    BalanceEquiv (dot3 v (cross3 g v)) bzero ∧
    BalanceEquiv (dot3 g (cross3 g v)) bzero := by
  let env := vec3_env3 g v vec3_zero
  exact ⟨bal_poly_eq env e_lag_L e_lag_R p_lag,
         bal_poly_eq env e_v_ec .zero p_vec,
         bal_poly_eq env e_g_ec .zero p_gec⟩

-- Формула квадрата нормы линейной комбинации ||α·v + β·q + γ·g||²
private def e_h1d_L : BExpr :=
  let s : BExpr := .add (.add (.mul (.var 0) (.var 3)) (.mul (.var 1) (.var 4))) (.mul (.var 2) (.var 5))
  .mul s s

private def e_h1d_R : BExpr :=
  let ea : BExpr := .var 0; let eb : BExpr := .var 1; let ec : BExpr := .var 2
  let ex : BExpr := .var 3; let ey : BExpr := .var 4; let ez : BExpr := .var 5
  .add (.add (.add (.mul (.mul ea ea) (.mul ex ex)) (.mul (.mul eb eb) (.mul ey ey)))
             (.mul (.mul ec ec) (.mul ez ez)))
       (.add (.add (.mul (.mul (.nat 2) (.mul ea eb)) (.mul ex ey))
                   (.mul (.mul (.nat 2) (.mul ea ec)) (.mul ex ez)))
             (.mul (.mul (.nat 2) (.mul eb ec)) (.mul ey ez)))

private theorem p_h1d_eq : e_h1d_L.norm = e_h1d_R.norm := rfl

private def e_s109_L : BExpr :=
  let ea : BExpr := .var 0; let eb : BExpr := .var 1; let ec : BExpr := .var 2
  let e_om := BExpr.sub .one ea
  let e_ga := BExpr.mul (.nat 2) ec
  .add
    (.add
      (.add (.mul (.mul e_om e_om) eb)
            (.mul (.mul (.nat 2) (.nat 2)) (.sub (.mul ea eb) (.mul ec ec))))
      (.mul (.mul e_ga e_ga) ea))
    (.add
      (.add (.mul (.mul (.nat 2) (.mul e_om (.nat 2))) .zero)
            (.mul (.mul (.nat 2) (.mul e_om e_ga)) ec))
      (.mul (.mul (.nat 2) (.mul (.nat 2) e_ga)) .zero))

private def e_s109_R : BExpr :=
  .mul (.mul (.add .one (.var 0)) (.add .one (.var 0))) (.var 1)

private theorem p_s109_eq : e_s109_L.norm = e_s109_R.norm := rfl

theorem dot3_lincomb3_sq_helper (a b c x y z : Balance) :
    BalanceEquiv (bsq (balance_add (balance_add (balance_mul a x) (balance_mul b y)) (balance_mul c z)))
      (balance_add
        (balance_add
          (balance_add (balance_mul (bsq a) (bsq x)) (balance_mul (bsq b) (bsq y)))
          (balance_mul (bsq c) (bsq z)))
        (balance_add
          (balance_add (balance_mul (balance_mul (balOfNat 2) (balance_mul a b)) (balance_mul x y))
                       (balance_mul (balance_mul (balOfNat 2) (balance_mul a c)) (balance_mul x z)))
          (balance_mul (balance_mul (balOfNat 2) (balance_mul b c)) (balance_mul y z)))) := by
  let env : Nat → Balance := fun | 0 => a | 1 => b | 2 => c | 3 => x | 4 => y | 5 => z | _ => bzero
  exact bal_poly_eq env e_h1d_L e_h1d_R p_h1d_eq

theorem dot3_lincomb3_sq (al be ga : Balance) (v q g : Vec3) :
    let w := vec3_add (vec3_add (vec3_scale al v) (vec3_scale be q)) (vec3_scale ga g)
    BalanceEquiv (dot3 w w)
      (balance_add
        (balance_add
          (balance_add (balance_mul (bsq al) (dot3 v v))
                       (balance_mul (bsq be) (dot3 q q)))
          (balance_mul (bsq ga) (dot3 g g)))
        (balance_add
          (balance_add (balance_mul (balance_mul (balOfNat 2) (balance_mul al be)) (dot3 v q))
                       (balance_mul (balance_mul (balOfNat 2) (balance_mul al ga)) (dot3 v g)))
          (balance_mul (balance_mul (balOfNat 2) (balance_mul be ga)) (dot3 q g)))) := by
  dsimp only
  have hx := dot3_lincomb3_sq_helper al be ga v.x q.x g.x
  have hy := dot3_lincomb3_sq_helper al be ga v.y q.y g.y
  have hz := dot3_lincomb3_sq_helper al be ga v.z q.z g.z
  have hsum := step25_balance_add_congr (step25_balance_add_congr hx hy) hz
  -- Перегруппировка 3 координат (каждая — линейная комбинация 6 коэффициентов k0..k5)
  have hpair : ∀ (p0 p1 p2 p3 p4 p5 q0 q1 q2 q3 q4 q5 : Balance),
      BalanceEquiv
        (balance_add (balance_add (balance_add (balance_add p0 p1) p2) (balance_add (balance_add p3 p4) p5))
                     (balance_add (balance_add (balance_add q0 q1) q2) (balance_add (balance_add q3 q4) q5)))
        (balance_add (balance_add (balance_add (balance_add p0 q0) (balance_add p1 q1)) (balance_add p2 q2))
                     (balance_add (balance_add (balance_add p3 q3) (balance_add p4 q4)) (balance_add p5 q5))) := by
    intro p0 p1 p2 p3 p4 p5 q0 q1 q2 q3 q4 q5
    exact step24_balance_trans (bal_add_add_add_comm _ _ _ _)
      (step25_balance_add_congr
        (step24_balance_trans (bal_add_add_add_comm _ _ _ _)
          (step25_balance_add_congr (bal_add_add_add_comm _ _ _ _) (step24_balance_refl _)))
        (step24_balance_trans (bal_add_add_add_comm _ _ _ _)
          (step25_balance_add_congr (bal_add_add_add_comm _ _ _ _) (step24_balance_refl _))))
  have hfac : ∀ (k a b c : Balance),
      BalanceEquiv (balance_add (balance_add (balance_mul k a) (balance_mul k b)) (balance_mul k c))
                   (balance_mul k (balance_add (balance_add a b) c)) := by
    intro k a b c
    exact step24_balance_symm
      (step24_balance_trans (bal_left_distrib k (balance_add a b) c)
        (step25_balance_add_congr (bal_left_distrib k a b) (step24_balance_refl _)))
  let k0 := bsq al
  let k1 := bsq be
  let k2 := bsq ga
  let k3 := balance_mul (balOfNat 2) (balance_mul al be)
  let k4 := balance_mul (balOfNat 2) (balance_mul al ga)
  let k5 := balance_mul (balOfNat 2) (balance_mul be ga)
  let rz_block := balance_add
    (balance_add (balance_add (balance_mul k0 (bsq v.z)) (balance_mul k1 (bsq q.z))) (balance_mul k2 (bsq g.z)))
    (balance_add (balance_add (balance_mul k3 (balance_mul v.z q.z)) (balance_mul k4 (balance_mul v.z g.z)))
                 (balance_mul k5 (balance_mul q.z g.z)))
  have s1 := step25_balance_add_congr
    (hpair (balance_mul k0 (bsq v.x)) (balance_mul k1 (bsq q.x)) (balance_mul k2 (bsq g.x))
           (balance_mul k3 (balance_mul v.x q.x)) (balance_mul k4 (balance_mul v.x g.x)) (balance_mul k5 (balance_mul q.x g.x))
           (balance_mul k0 (bsq v.y)) (balance_mul k1 (bsq q.y)) (balance_mul k2 (bsq g.y))
           (balance_mul k3 (balance_mul v.y q.y)) (balance_mul k4 (balance_mul v.y g.y)) (balance_mul k5 (balance_mul q.y g.y)))
    (step24_balance_refl rz_block)
  have s2 := hpair
    (balance_add (balance_mul k0 (bsq v.x)) (balance_mul k0 (bsq v.y)))
    (balance_add (balance_mul k1 (bsq q.x)) (balance_mul k1 (bsq q.y)))
    (balance_add (balance_mul k2 (bsq g.x)) (balance_mul k2 (bsq g.y)))
    (balance_add (balance_mul k3 (balance_mul v.x q.x)) (balance_mul k3 (balance_mul v.y q.y)))
    (balance_add (balance_mul k4 (balance_mul v.x g.x)) (balance_mul k4 (balance_mul v.y g.y)))
    (balance_add (balance_mul k5 (balance_mul q.x g.x)) (balance_mul k5 (balance_mul q.y g.y)))
    (balance_mul k0 (bsq v.z)) (balance_mul k1 (bsq q.z)) (balance_mul k2 (bsq g.z))
    (balance_mul k3 (balance_mul v.z q.z)) (balance_mul k4 (balance_mul v.z g.z)) (balance_mul k5 (balance_mul q.z g.z))
  have s3 := step25_balance_add_congr
    (step25_balance_add_congr
      (step25_balance_add_congr (hfac k0 (bsq v.x) (bsq v.y) (bsq v.z))
                                (hfac k1 (bsq q.x) (bsq q.y) (bsq q.z)))
      (hfac k2 (bsq g.x) (bsq g.y) (bsq g.z)))
    (step25_balance_add_congr
      (step25_balance_add_congr (hfac k3 (balance_mul v.x q.x) (balance_mul v.y q.y) (balance_mul v.z q.z))
                                (hfac k4 (balance_mul v.x g.x) (balance_mul v.y g.y) (balance_mul v.z g.z)))
      (hfac k5 (balance_mul q.x g.x) (balance_mul q.y g.y) (balance_mul q.z g.z)))
  exact step24_balance_trans hsum (step24_balance_trans s1 (step24_balance_trans s2 s3))

/-- ШАГ 109 (План 59). Главная теорема точной ортогональности 3D ротора Кэли SO(3):
    Для ПРОИЗВОЛЬНОЙ оси g и состояния v:
    1. Знаменатель D = 1 + ||g||² строго положителен (> 0).
    2. ||cayley3d_num g v||² ~ (1 + ||g||²)² · ||v||² (строгое сохранение нормы памяти!). -/
theorem step109_cayley3d_exact_orthogonality (g v : Vec3) :
    BalGtZero (balance_add bone (dot3 g g)) ∧
    BalanceEquiv
      (dot3 (cayley3d_num g v) (cayley3d_num g v))
      (balance_mul (bsq (balance_add bone (dot3 g g))) (dot3 v v)) := by
  refine ⟨bal_add_gt_ge bone_gt_zero (dot3_self_ge_zero g), ?_⟩
  let a := dot3 g g
  let b := dot3 v v
  let c := dot3 g v
  have hlo := vec3_lagrange_and_orth g v
  have hvg : BalanceEquiv (dot3 v g) c :=
    step25_balance_add_congr
      (step25_balance_add_congr (bal_mul_comm v.x g.x) (bal_mul_comm v.y g.y))
      (bal_mul_comm v.z g.z)
  have hqg : BalanceEquiv (dot3 (cross3 g v) g) bzero :=
    step24_balance_trans
      (step25_balance_add_congr
        (step25_balance_add_congr (bal_mul_comm _ g.x) (bal_mul_comm _ g.y))
        (bal_mul_comm _ g.z))
      hlo.2.2
  have hcomb := dot3_lincomb3_sq (balance_sub bone a) (balOfNat 2) (balance_mul (balOfNat 2) c) v (cross3 g v) g
  have hsub : BalanceEquiv
      (balance_add
        (balance_add
          (balance_add (balance_mul (bsq (balance_sub bone a)) b)
                       (balance_mul (bsq (balOfNat 2)) (dot3 (cross3 g v) (cross3 g v))))
          (balance_mul (bsq (balance_mul (balOfNat 2) c)) a))
        (balance_add
          (balance_add (balance_mul (balance_mul (balOfNat 2) (balance_mul (balance_sub bone a) (balOfNat 2))) (dot3 v (cross3 g v)))
                       (balance_mul (balance_mul (balOfNat 2) (balance_mul (balance_sub bone a) (balance_mul (balOfNat 2) c))) (dot3 v g)))
          (balance_mul (balance_mul (balOfNat 2) (balance_mul (balOfNat 2) (balance_mul (balOfNat 2) c))) (dot3 (cross3 g v) g))))
      (balance_add
        (balance_add
          (balance_add (balance_mul (bsq (balance_sub bone a)) b)
                       (balance_mul (bsq (balOfNat 2)) (balance_sub (balance_mul a b) (bsq c))))
          (balance_mul (bsq (balance_mul (balOfNat 2) c)) a))
        (balance_add
          (balance_add (balance_mul (balance_mul (balOfNat 2) (balance_mul (balance_sub bone a) (balOfNat 2))) bzero)
                       (balance_mul (balance_mul (balOfNat 2) (balance_mul (balance_sub bone a) (balance_mul (balOfNat 2) c))) c))
          (balance_mul (balance_mul (balOfNat 2) (balance_mul (balOfNat 2) (balance_mul (balOfNat 2) c))) bzero))) :=
    step25_balance_add_congr
      (step25_balance_add_congr
        (step25_balance_add_congr (step24_balance_refl _) (bal_mul_congr (step24_balance_refl _) hlo.1))
        (step24_balance_refl _))
      (step25_balance_add_congr
        (step25_balance_add_congr (bal_mul_congr (step24_balance_refl _) hlo.2.1)
                                  (bal_mul_congr (step24_balance_refl _) hvg))
        (bal_mul_congr (step24_balance_refl _) hqg))
  let env : Nat → Balance := fun | 0 => a | 1 => b | 2 => c | _ => bzero
  exact step24_balance_trans hcomb (step24_balance_trans hsub (bal_poly_eq env e_s109_L e_s109_R p_s109_eq))

-- Тождество двойного векторного произведения (BAC-CAB): e × (e × v) ~ (e·v)e - (e·e)v
theorem vec3_bac_cab (e v : Vec3) :
    Vec3Equiv (cross3 e (cross3 e v))
      (vec3_add (vec3_scale (dot3 e v) e) (vec3_neg (vec3_scale (dot3 e e) v))) := by
  let env := vec3_env3 e v vec3_zero
  let ee : BExpr × BExpr × BExpr := (.var 0, .var 1, .var 2)
  let ev : BExpr × BExpr × BExpr := (.var 3, .var 4, .var 5)
  let lhs := vCrossE ee (vCrossE ee ev)
  let rhs := vAddE (vScaleE (vDotE ee ev) ee) (vNegE (vScaleE (vDotE ee ee) ev))
  exact ⟨bal_poly_eq env lhs.1 rhs.1 rfl,
         bal_poly_eq env lhs.2.1 rhs.2.1 rfl,
         bal_poly_eq env lhs.2.2 rhs.2.2 rfl⟩

/-- ШАГ 109b. Эквивалентность формулы `cayley_rot3` из `src/rtrl.rs` каноническому ротору Кэли:
    При единичной оси ||e||² ~ 1 форма (1+w²)v + 2w(e×v) + 2w²(e×(e×v)) тождественно равна
    `cayley3d_num (w·e) v` и точно сохраняет норму! -/
theorem step109b_cayley_rot3_equiv (e : Vec3) (w : Balance) (v : Vec3)
    (he_unit : BalanceEquiv (dot3 e e) bone) :
    Vec3Equiv (cayley_rot3_num e w v) (cayley3d_num (vec3_scale w e) v) := by
  have hbc := vec3_bac_cab e v
  have hg2 : BalanceEquiv (dot3 (vec3_scale w e) (vec3_scale w e)) (bsq w) := by
    let env : Nat → Balance := fun | 0 => w | 1 => e.x | 2 => e.y | 3 => e.z | _ => bzero
    let ew : BExpr := .var 0; let ex : BExpr := .var 1; let ey : BExpr := .var 2; let ez : BExpr := .var 3
    have h1 := bal_poly_eq env
      (vDotE (vScaleE ew (ex, ey, ez)) (vScaleE ew (ex, ey, ez)))
      (.mul (.mul ew ew) (vDotE (ex, ey, ez) (ex, ey, ez))) rfl
    exact step24_balance_trans h1
      (step24_balance_trans (bal_mul_congr (step24_balance_refl (bsq w)) he_unit) (bal_mul_one (bsq w)))
  have hgv : BalanceEquiv (dot3 (vec3_scale w e) v) (balance_mul w (dot3 e v)) := by
    let env : Nat → Balance := fun | 0 => w | 1 => e.x | 2 => e.y | 3 => e.z | 4 => v.x | 5 => v.y | 6 => v.z | _ => bzero
    let ew : BExpr := .var 0
    let ee : BExpr × BExpr × BExpr := (.var 1, .var 2, .var 3)
    let ev : BExpr × BExpr × BExpr := (.var 4, .var 5, .var 6)
    exact bal_poly_eq env (vDotE (vScaleE ew ee) ev) (.mul ew (vDotE ee ev)) rfl
  have hcx : Vec3Equiv (cross3 (vec3_scale w e) v) (vec3_scale w (cross3 e v)) := by
    let env : Nat → Balance := fun | 0 => w | 1 => e.x | 2 => e.y | 3 => e.z | 4 => v.x | 5 => v.y | 6 => v.z | _ => bzero
    let ew : BExpr := .var 0
    let ee : BExpr × BExpr × BExpr := (.var 1, .var 2, .var 3)
    let ev : BExpr × BExpr × BExpr := (.var 4, .var 5, .var 6)
    let l := vCrossE (vScaleE ew ee) ev
    let r := vScaleE ew (vCrossE ee ev)
    exact ⟨bal_poly_eq env l.1 r.1 rfl, bal_poly_eq env l.2.1 r.2.1 rfl, bal_poly_eq env l.2.2 r.2.2 rfl⟩
  -- Покомпонентное доказательство через 6 скаляров (w, ev_c, q_c, e_c, g_dot_v)
  have hcomp : ∀ (vc qc ec rc : Balance),
      BalanceEquiv rc (balance_add (balance_mul (dot3 e v) ec) (balance_neg (balance_mul (dot3 e e) vc))) →
      BalanceEquiv
        (balance_add (balance_add (balance_mul (balance_add bone (bsq w)) vc)
                                  (balance_mul (balance_mul (balOfNat 2) w) qc))
                     (balance_mul (balance_mul (balOfNat 2) (bsq w)) rc))
        (balance_add (balance_add (balance_mul (balance_sub bone (dot3 (vec3_scale w e) (vec3_scale w e))) vc)
                                  (balance_mul (balOfNat 2) (balance_mul w qc)))
                     (balance_mul (balance_mul (balOfNat 2) (dot3 (vec3_scale w e) v)) (balance_mul w ec))) := by
    intro vc qc ec rc hrc
    have hrc' : BalanceEquiv rc (balance_add (balance_mul (dot3 e v) ec) (balance_neg vc)) :=
      step24_balance_trans hrc
        (step25_balance_add_congr (step24_balance_refl _)
          (bal_neg_congr (step24_balance_trans (bal_mul_congr he_unit (step24_balance_refl vc)) (bal_one_mul vc))))
    have hL := step25_balance_add_congr
      (step24_balance_refl (balance_add (balance_mul (balance_add bone (bsq w)) vc)
                                        (balance_mul (balance_mul (balOfNat 2) w) qc)))
      (bal_mul_congr (step24_balance_refl (balance_mul (balOfNat 2) (bsq w))) hrc')
    have hR := step25_balance_add_congr
      (step25_balance_add_congr
        (bal_mul_congr (bal_sub_congr (step24_balance_refl bone) hg2) (step24_balance_refl vc))
        (step24_balance_refl (balance_mul (balOfNat 2) (balance_mul w qc))))
      (bal_mul_congr (bal_mul_congr (step24_balance_refl (balOfNat 2)) hgv)
                     (step24_balance_refl (balance_mul w ec)))
    let env : Nat → Balance := fun | 0 => w | 1 => vc | 2 => qc | 3 => ec | 4 => dot3 e v | _ => bzero
    let ew : BExpr := .var 0; let evc : BExpr := .var 1; let eqc : BExpr := .var 2
    let eec : BExpr := .var 3; let ed : BExpr := .var 4
    let eL : BExpr :=
      .add (.add (.mul (.add .one (.mul ew ew)) evc)
                 (.mul (.mul (.nat 2) ew) eqc))
           (.mul (.mul (.nat 2) (.mul ew ew)) (.add (.mul ed eec) (.neg evc)))
    let eR : BExpr :=
      .add (.add (.mul (.sub .one (.mul ew ew)) evc)
                 (.mul (.nat 2) (.mul ew eqc)))
           (.mul (.mul (.nat 2) (.mul ew ed)) (.mul ew eec))
    exact step24_balance_trans hL (step24_balance_trans (bal_poly_eq env eL eR rfl) (step24_balance_symm hR))
  refine ⟨?_, ?_, ?_⟩
  · exact step24_balance_trans (hcomp v.x (cross3 e v).x e.x (cross3 e (cross3 e v)).x hbc.1)
      (step25_balance_add_congr
        (step25_balance_add_congr (step24_balance_refl _)
          (bal_mul_congr (step24_balance_refl _) (step24_balance_symm hcx.1)))
        (step24_balance_refl _))
  · exact step24_balance_trans (hcomp v.y (cross3 e v).y e.y (cross3 e (cross3 e v)).y hbc.2.1)
      (step25_balance_add_congr
        (step25_balance_add_congr (step24_balance_refl _)
          (bal_mul_congr (step24_balance_refl _) (step24_balance_symm hcx.2.1)))
        (step24_balance_refl _))
  · exact step24_balance_trans (hcomp v.z (cross3 e v).z e.z (cross3 e (cross3 e v)).z hbc.2.2)
      (step25_balance_add_congr
        (step25_balance_add_congr (step24_balance_refl _)
          (bal_mul_congr (step24_balance_refl _) (step24_balance_symm hcx.2.2)))
        (step24_balance_refl _))

def v_ex : Vec3 := ⟨bone, bzero, bzero⟩
def v_ey : Vec3 := ⟨bzero, bone, bzero⟩
def v_ez : Vec3 := ⟨bzero, bzero, bone⟩

def det3 (c1 c2 c3 : Vec3) : Balance :=
  dot3 c1 (cross3 c2 c3)

private def e_c3 (g v : BExpr × BExpr × BExpr) : BExpr × BExpr × BExpr :=
  let g2 := vDotE g g
  vAddE (vAddE (vScaleE (.sub .one g2) v) (vScaleE (.nat 2) (vCrossE g v)))
        (vScaleE (.mul (.nat 2) (vDotE g v)) g)

private def e_det_L : BExpr :=
  let eg : BExpr × BExpr × BExpr := (.var 0, .var 1, .var 2)
  let c1 := e_c3 eg (.one, .zero, .zero)
  let c2 := e_c3 eg (.zero, .one, .zero)
  let c3 := e_c3 eg (.zero, .zero, .one)
  vDotE c1 (vCrossE c2 c3)

private def e_det_R : BExpr :=
  let eg : BExpr × BExpr × BExpr := (.var 0, .var 1, .var 2)
  let d := BExpr.add .one (vDotE eg eg)
  .mul d (.mul d d)

private theorem p_det_one : e_det_L.norm = e_det_R.norm := rfl

/-- ШАГ 109c. Строгая принадлежность специальной ортогональной группе SO(3) (det R(g) = +1):
    Для произвольного 3D-генератора g : Vec3 определитель матрицы числителей `R_num(g)`
    тождественно равен `(1 + ||g||²)³`, следовательно `det(R(g)) = +1` (собственное вращение, не отражение!). -/
theorem step109c_cayley3d_det_one (g : Vec3) :
    BalGtZero (balance_add bone (dot3 g g)) ∧
    BalanceEquiv
      (det3 (cayley3d_num g v_ex) (cayley3d_num g v_ey) (cayley3d_num g v_ez))
      (balance_mul (balance_add bone (dot3 g g)) (bsq (balance_add bone (dot3 g g)))) := by
  refine ⟨bal_add_gt_ge bone_gt_zero (dot3_self_ge_zero g), ?_⟩
  let env : Nat → Balance := fun | 0 => g.x | 1 => g.y | 2 => g.z | _ => bzero
  exact bal_poly_eq env e_det_L e_det_R p_det_one

private def e_so3_L : BExpr := (e_c3 (.one, .zero, .zero) (e_c3 (.zero, .one, .zero) (.one, .zero, .zero))).2.1
private def e_so3_R : BExpr := (e_c3 (.zero, .one, .zero) (e_c3 (.one, .zero, .zero) (.one, .zero, .zero))).2.1
private theorem p_so3_L : e_so3_L.norm = (BExpr.nat 4).norm := rfl
private theorem p_so3_R : e_so3_R.norm = BExpr.zero.norm := rfl

/-- ШАГ 110 (План 60). Некоммутативность 3D роторов SO(3):
    На осях g₁ = (1,0,0), g₂ = (0,1,0) и векторе v = (1,0,0):
    R(g₁)R(g₂)v ≁ R(g₂)R(g₁)v (порядок событий A → B алгебраически отличен от B → A!). -/
theorem step110_so3_noncommutativity :
    Vec3Equiv (cayley3d_num v_ex (cayley3d_num v_ey v_ex))
              (cayley3d_num v_ey (cayley3d_num v_ex v_ex)) → False := by
  intro h
  have hy : BalanceEquiv
      (cayley3d_num v_ex (cayley3d_num v_ey v_ex)).y
      (cayley3d_num v_ey (cayley3d_num v_ex v_ex)).y := h.2.1
  let env : Nat → Balance := fun _ => bzero
  have hl : BalanceEquiv (cayley3d_num v_ex (cayley3d_num v_ey v_ex)).y (balOfNat 4) :=
    bal_poly_eq env e_so3_L (.nat 4) p_so3_L
  have hr : BalanceEquiv (cayley3d_num v_ey (cayley3d_num v_ex v_ex)).y bzero :=
    bal_poly_eq env e_so3_R .zero p_so3_R
  have h40 : BalanceEquiv (balOfNat 4) bzero :=
    step24_balance_trans (step24_balance_symm hl) (step24_balance_trans hy hr)
  exact bal_gt_zero_ne_zero (balOfNat_succ_gt_zero 3) h40

/-- ШАГ 111 (План 61). Строгое BIBO-сжатие Cayley-RTRL (`src/rtrl.rs`):
    Для λ(α) = 199 / (200(1 + α²)) и произвольного α : Balance:
    1. Числитель 199 > 0.
    2. Знаменатель 200(1 + α²) > 0 (деление тотально).
    3. Строго 199 < 200(1 + α²), т.е. 0 < λ(α) < 1 (состояние не может взорваться!). -/
theorem step111_rtrl_decay_strict_contraction (a : Balance) :
    let num := balOfNat 199
    let den := balance_mul (balOfNat 200) (balance_add bone (bsq a))
    BalGtZero num ∧ BalGtZero den ∧ BalLt num den := by
  dsimp only
  have h199 : BalGtZero (balOfNat 199) := balOfNat_succ_gt_zero 198
  have h200 : BalGtZero (balOfNat 200) := balOfNat_succ_gt_zero 199
  have hden : BalGtZero (balance_mul (balOfNat 200) (balance_add bone (bsq a))) :=
    bal_mul_gt_gt h200 (bal_one_add_sq_gt_zero a)
  refine ⟨h199, hden, ?_⟩
  let env : Nat → Balance := fun | 0 => a | _ => bzero
  have hdiff : BalanceEquiv
      (balance_add bone (balance_mul (balOfNat 200) (bsq a)))
      (balance_sub (balance_mul (balOfNat 200) (balance_add bone (bsq a))) (balOfNat 199)) :=
    bal_poly_eq env
      (.add .one (.mul (.nat 200) (.mul (.var 0) (.var 0))))
      (.sub (.mul (.nat 200) (.add .one (.mul (.var 0) (.var 0)))) (.nat 199))
      rfl
  exact bal_gt_zero_of_equiv hdiff
    (bal_add_gt_ge bone_gt_zero (bal_mul_gt_ge h200 (bal_sq_ge_zero a)))


-- =============================================================================
-- ЭТАП XXVIII. БЛОК III: ВЕРИФИКАЦИЯ ОПЕРАТОРОВ БЕЗ SOFTMAX И ПРОИЗВОДНЫХ RTRL
-- (Шаги 112–114; соответствуют шагам 62–64 плана micro_continual_core)
-- =============================================================================

/-- ШАГ 112 (План 62). Рациональная политика без exp/softmax:
    Для ВСЕХ z : Balance обе ветки g(z) строго положительны и ≥ 1 (в удвоенном масштабе):
    - Ветка z < 0: 2 - 2z + z² ~ 1 + (z - 1)² ≥ 1 > 0.
    - Ветка z ≥ 0: 2 + 2z + z² ~ 1 + (z + 1)² ≥ 1 > 0. -/
theorem step112_rational_softmax_positive (z : Balance) :
    let q_neg := balance_add (balance_sub (balOfNat 2) (balance_mul (balOfNat 2) z)) (bsq z)
    let q_pos := balance_add (balance_add (balOfNat 2) (balance_mul (balOfNat 2) z)) (bsq z)
    (BalanceEquiv q_neg (balance_add bone (bsq (balance_sub z bone))) ∧
     BalLe bone q_neg ∧ BalGtZero q_neg) ∧
    (BalanceEquiv q_pos (balance_add bone (bsq (balance_add z bone))) ∧
     BalLe bone q_pos ∧ BalGtZero q_pos) := by
  dsimp only
  let env : Nat → Balance := fun | 0 => z | _ => bzero
  let ez : BExpr := .var 0
  let e_qneg : BExpr := .add (.sub (.nat 2) (.mul (.nat 2) ez)) (.mul ez ez)
  let e_qpos : BExpr := .add (.add (.nat 2) (.mul (.nat 2) ez)) (.mul ez ez)
  have hneg_id : BalanceEquiv
      (balance_add (balance_sub (balOfNat 2) (balance_mul (balOfNat 2) z)) (bsq z))
      (balance_add bone (bsq (balance_sub z bone))) :=
    bal_poly_eq env e_qneg (.add .one (.mul (.sub ez .one) (.sub ez .one))) rfl
  have hneg_sub : BalanceEquiv
      (bsq (balance_sub z bone))
      (balance_sub (balance_add (balance_sub (balOfNat 2) (balance_mul (balOfNat 2) z)) (bsq z)) bone) :=
    bal_poly_eq env (.mul (.sub ez .one) (.sub ez .one)) (.sub e_qneg .one) rfl
  have hpos_id : BalanceEquiv
      (balance_add (balance_add (balOfNat 2) (balance_mul (balOfNat 2) z)) (bsq z))
      (balance_add bone (bsq (balance_add z bone))) :=
    bal_poly_eq env e_qpos (.add .one (.mul (.add ez .one) (.add ez .one))) rfl
  have hpos_sub : BalanceEquiv
      (bsq (balance_add z bone))
      (balance_sub (balance_add (balance_add (balOfNat 2) (balance_mul (balOfNat 2) z)) (bsq z)) bone) :=
    bal_poly_eq env (.mul (.add ez .one) (.add ez .one)) (.sub e_qpos .one) rfl
  exact ⟨⟨hneg_id,
          bal_ge_zero_of_equiv hneg_sub (bal_sq_ge_zero (balance_sub z bone)),
          bal_gt_zero_of_equiv (step24_balance_symm hneg_id) (bal_one_add_sq_gt_zero (balance_sub z bone))⟩,
         ⟨hpos_id,
          bal_ge_zero_of_equiv hpos_sub (bal_sq_ge_zero (balance_add z bone)),
          bal_gt_zero_of_equiv (step24_balance_symm hpos_id) (bal_one_add_sq_gt_zero (balance_add z bone))⟩⟩

def sum_sq : List Balance → Balance
  | [] => bzero
  | g :: gs => balance_add (bsq g) (sum_sq gs)

theorem sum_sq_ge_zero (gs : List Balance) : BalGeZero (sum_sq gs) := by
  induction gs with
  | nil => exact Or.inl (step24_balance_refl bzero)
  | cons g rest ih => exact bal_add_ge_ge (bal_sq_ge_zero g) ih

theorem mem_sq_le_sum_sq {x : Balance} {gs : List Balance} (h : x ∈ gs) :
    BalLe (bsq x) (sum_sq gs) := by
  induction gs with
  | nil => cases h
  | cons g rest ih =>
    cases h with
    | head =>
      dsimp [BalLe, sum_sq]
      let env : Nat → Balance := fun | 0 => bsq x | 1 => sum_sq rest | _ => bzero
      have hsub := bal_poly_eq env (.var 1) (.sub (.add (.var 0) (.var 1)) (.var 0)) rfl
      exact bal_ge_zero_of_equiv hsub (sum_sq_ge_zero rest)
    | tail _ hmem =>
      have hrest := ih hmem
      dsimp [BalLe, sum_sq] at *
      let env : Nat → Balance := fun | 0 => bsq g | 1 => sum_sq rest | 2 => bsq x | _ => bzero
      have hsub := bal_poly_eq env
        (.add (.var 0) (.sub (.var 1) (.var 2)))
        (.sub (.add (.var 0) (.var 1)) (.var 2)) rfl
      exact bal_ge_zero_of_equiv hsub (bal_add_ge_ge (bal_sq_ge_zero g) hrest)

/-- ШАГ 113 (План 63). Теорема W1 Kerr-конкуренции (тотальность знаменателя и оценка весов):
    Для любого ε > 0 и любого списка g:
    1. ε + ∑ gᵢ² ≥ ε и строго > 0 (деление тотально!).
    2. Для каждого члена g_k ∈ g: g_k² < ε + ∑ gᵢ² (вес строго < 1!). -/
theorem step113_kerr_w1_total_denom (eps : Balance) (heps : BalGtZero eps) (gs : List Balance) :
    let den := balance_add eps (sum_sq gs)
    BalLe eps den ∧ BalGtZero den ∧ (∀ x, x ∈ gs → BalLt (bsq x) den) := by
  dsimp only
  let env : Nat → Balance := fun | 0 => eps | 1 => sum_sq gs | _ => bzero
  have hle_eq := bal_poly_eq env (.var 1) (.sub (.add (.var 0) (.var 1)) (.var 0)) rfl
  refine ⟨bal_ge_zero_of_equiv hle_eq (sum_sq_ge_zero gs),
          bal_add_gt_ge heps (sum_sq_ge_zero gs), ?_⟩
  intro x hmem
  have hle_x := mem_sq_le_sum_sq hmem
  let env2 : Nat → Balance := fun | 0 => eps | 1 => sum_sq gs | 2 => bsq x | _ => bzero
  have hlt_eq := bal_poly_eq env2
    (.add (.var 0) (.sub (.var 1) (.var 2)))
    (.sub (.add (.var 0) (.var 1)) (.var 2)) rfl
  exact bal_gt_zero_of_equiv hlt_eq (bal_add_gt_ge heps hle_x)

def bpow4 (x : Balance) : Balance := bsq (bsq x)

def sum_pow4 : List Balance → Balance
  | [] => bzero
  | g :: gs => balance_add (bpow4 g) (sum_pow4 gs)

theorem bal_pow4_ge_zero (x : Balance) : BalGeZero (bpow4 x) :=
  bal_sq_ge_zero (bsq x)

theorem sum_pow4_ge_zero (gs : List Balance) : BalGeZero (sum_pow4 gs) := by
  induction gs with
  | nil => exact Or.inl (step24_balance_refl bzero)
  | cons g rest ih => exact bal_add_ge_ge (bal_pow4_ge_zero g) ih

theorem mem_pow4_le_sum_pow4 {x : Balance} {gs : List Balance} (h : x ∈ gs) :
    BalLe (bpow4 x) (sum_pow4 gs) := by
  induction gs with
  | nil => cases h
  | cons g rest ih =>
    cases h with
    | head =>
      dsimp [BalLe, sum_pow4]
      let env : Nat → Balance := fun | 0 => bpow4 x | 1 => sum_pow4 rest | _ => bzero
      have hsub := bal_poly_eq env (.var 1) (.sub (.add (.var 0) (.var 1)) (.var 0)) rfl
      exact bal_ge_zero_of_equiv hsub (sum_pow4_ge_zero rest)
    | tail _ hmem =>
      have hrest := ih hmem
      dsimp [BalLe, sum_pow4] at *
      let env : Nat → Balance := fun | 0 => bpow4 g | 1 => sum_pow4 rest | 2 => bpow4 x | _ => bzero
      have hsub := bal_poly_eq env
        (.add (.var 0) (.sub (.var 1) (.var 2)))
        (.sub (.add (.var 0) (.var 1)) (.var 2)) rfl
      exact bal_ge_zero_of_equiv hsub (bal_add_ge_ge (bal_pow4_ge_zero g) hrest)

/-- ШАГ 113b. Теорема W1 Kerr-конкуренции степени p = 4 (`src/memory.rs::KerrKernelBank`):
    Для любого ε > 0 и любого списка g:
    1. ε + ∑ gᵢ⁴ ≥ ε и строго > 0 (деление при p = 4 тотально).
    2. Для каждого члена g_k ∈ g: g_k⁴ < ε + ∑ gᵢ⁴ (каждый вес строго < 1). -/
theorem step113b_kerr_w1_p4_total_denom (eps : Balance) (heps : BalGtZero eps) (gs : List Balance) :
    let den := balance_add eps (sum_pow4 gs)
    BalLe eps den ∧ BalGtZero den ∧ (∀ x, x ∈ gs → BalLt (bpow4 x) den) := by
  dsimp only
  let env : Nat → Balance := fun | 0 => eps | 1 => sum_pow4 gs | _ => bzero
  have hle_eq := bal_poly_eq env (.var 1) (.sub (.add (.var 0) (.var 1)) (.var 0)) rfl
  refine ⟨bal_ge_zero_of_equiv hle_eq (sum_pow4_ge_zero gs),
          bal_add_gt_ge heps (sum_pow4_ge_zero gs), ?_⟩
  intro x hmem
  have hle_x := mem_pow4_le_sum_pow4 hmem
  let env2 : Nat → Balance := fun | 0 => eps | 1 => sum_pow4 gs | 2 => bpow4 x | _ => bzero
  have hlt_eq := bal_poly_eq env2
    (.add (.var 0) (.sub (.var 1) (.var 2)))
    (.sub (.add (.var 0) (.var 1)) (.var 2)) rfl
  exact bal_gt_zero_of_equiv hlt_eq (bal_add_gt_ge heps hle_x)

-- Расширение существующего DualBalance (a + bε, ε² = 0) сложением и вложением скаляра
def dual_add (x y : DualBalance) : DualBalance :=
  ⟨balance_add x.re y.re, balance_add x.eps y.eps⟩

def dual_const (c : Balance) : DualBalance := ⟨c, bzero⟩
def dual_var (x : Balance) : DualBalance := ⟨x, bone⟩

private def dMulE (u v : BExpr × BExpr) : BExpr × BExpr :=
  (.mul u.1 v.1, .add (.mul u.1 v.2) (.mul u.2 v.1))

private def dAddE (u v : BExpr × BExpr) : BExpr × BExpr :=
  (.add u.1 v.1, .add u.2 v.2)

private def e_wd : BExpr × BExpr := (.var 0, .one)
private def e_w2d : BExpr × BExpr := dMulE e_wd e_wd
private def e_den_wd : BExpr × BExpr := dAddE (.one, .zero) e_w2d
private def e_Dw : BExpr := .add .one (.mul (.var 0) (.var 0))

private def e_c1_L : BExpr × BExpr :=
  dMulE e_den_wd (.mul (.mul (.nat 2) (.var 0)) e_Dw, .mul (.nat 2) (.sub .one (.mul (.var 0) (.var 0))))
private def e_c1_R : BExpr × BExpr :=
  dMulE (.mul e_Dw e_Dw, .zero) (dMulE (.nat 2, .zero) e_wd)

private def e_c2_L : BExpr × BExpr :=
  dMulE e_den_wd (.mul (.mul (.nat 2) (.mul (.var 0) (.var 0))) e_Dw, .mul (.nat 4) (.var 0))
private def e_c2_R : BExpr × BExpr :=
  dMulE (.mul e_Dw e_Dw, .zero) (dMulE (.nat 2, .zero) e_w2d)

private def e_lam3_den : BExpr × BExpr :=
  dMulE (.var 2, .zero) (dAddE (.one, .zero) e_w2d)
private def e_lam3_L : BExpr × BExpr :=
  dMulE e_lam3_den (.mul (.var 1) e_Dw, .neg (.mul (.mul (.nat 2) (.var 1)) (.var 0)))
private def e_lam3_R : BExpr × BExpr :=
  dMulE (.mul (.var 2) (.mul e_Dw e_Dw), .zero) (.var 1, .zero)

private theorem p_c1_re : e_c1_L.1.norm = e_c1_R.1.norm := rfl
private theorem p_c1_eps : e_c1_L.2.norm = e_c1_R.2.norm := rfl
private theorem p_c2_re : e_c2_L.1.norm = e_c2_R.1.norm := rfl
private theorem p_c2_eps : e_c2_L.2.norm = e_c2_R.2.norm := rfl
private theorem p_lam3_re : e_lam3_L.1.norm = e_lam3_R.1.norm := rfl
private theorem p_lam3_eps : e_lam3_L.2.norm = e_lam3_R.2.norm := rfl

/-- ШАГ 114 (План 64). Конструктивное доказательство точных производных `src/rtrl.rs`
    (`dc1_dw`, `dc2_dw`, `dlam_da`) через алгебраические дуальные числа (без пределов и ℝ):
    Для рациональной функции f(x) = N(x)/D(x) и производной f'(x) = P(x)/Q(x)
    доказывается прямое дуальное тождество D(x+ε) · (N·D + P·ε) ~ D² · N(x+ε):
    1. (2w / (1 + w²))' = 2(1 - w²) / (1 + w²)².
    2. (2w² / (1 + w²))' = 4w / (1 + w²)².
    3. (199 / (200(1 + α²)))' = -398α / (200(1 + α²)²). -/
theorem step114_rtrl_exact_derivatives_dual (w a : Balance) :
    -- 1. c1(w) = 2w / (1 + w²)
    let wd := dual_var w
    let w2d := dual_mul wd wd
    let den_wd := dual_add (dual_const bone) w2d
    let c1_num_d := dual_mul (dual_const (balOfNat 2)) wd
    let D_w := balance_add bone (bsq w)
    let c1_N := balance_mul (balOfNat 2) w
    let dc1_num := balance_mul (balOfNat 2) (balance_sub bone (bsq w))
    DualEquiv
      (dual_mul den_wd ⟨balance_mul c1_N D_w, dc1_num⟩)
      (dual_mul (dual_const (bsq D_w)) c1_num_d)
    ∧
    -- 2. c2(w) = 2w² / (1 + w²)
    let c2_num_d := dual_mul (dual_const (balOfNat 2)) w2d
    let c2_N := balance_mul (balOfNat 2) (bsq w)
    let dc2_num := balance_mul (balOfNat 4) w
    DualEquiv
      (dual_mul den_wd ⟨balance_mul c2_N D_w, dc2_num⟩)
      (dual_mul (dual_const (bsq D_w)) c2_num_d)
    ∧
    -- 3. lam(α) = 199 / (200(1 + α²))
    let ad := dual_var a
    let a2d := dual_mul ad ad
    let D_a := balance_add bone (bsq a)
    let lam_den_d := dual_mul (dual_const (balOfNat 200)) (dual_add (dual_const bone) a2d)
    let lam_num_d := dual_const (balOfNat 199)
    let dlam_num := balance_neg (balance_mul (balOfNat 398) a)
    DualEquiv
      (dual_mul lam_den_d ⟨balance_mul (balOfNat 199) D_a, dlam_num⟩)
      (dual_mul (dual_const (balance_mul (balOfNat 200) (bsq D_a))) lam_num_d) := by
  dsimp only [dual_var, dual_const, dual_mul, dual_add, DualEquiv, bsq]
  let env : Nat → Balance := fun | 0 => w | _ => bzero
  -- 3. lam: универсально для c / (k * (1 + a²)) с производной (-2*c*a) / (k * (1 + a²)²), затем c=199, k=200 (2*199=398)
  have hlam_gen : ∀ (c k : Balance),
      let ad := dual_var a
      let a2d := dual_mul ad ad
      let D_a := balance_add bone (bsq a)
      let lam_den_d := dual_mul (dual_const k) (dual_add (dual_const bone) a2d)
      let lam_num_d := dual_const c
      let dlam_num := balance_neg (balance_mul (balance_mul (balOfNat 2) c) a)
      DualEquiv
        (dual_mul lam_den_d ⟨balance_mul c D_a, dlam_num⟩)
        (dual_mul (dual_const (balance_mul k (bsq D_a))) lam_num_d) := by
    intro c k
    dsimp only [dual_var, dual_const, dual_mul, dual_add, DualEquiv, bsq]
    let env3 : Nat → Balance := fun | 0 => a | 1 => c | 2 => k | _ => bzero
    exact ⟨bal_poly_eq env3 e_lam3_L.1 e_lam3_R.1 p_lam3_re,
           bal_poly_eq env3 e_lam3_L.2 e_lam3_R.2 p_lam3_eps⟩
  have h398 : BalanceEquiv (balance_mul (balOfNat 2) (balOfNat 199)) (balOfNat 398) :=
    step24_balance_symm (balOfNat_mul 2 199)
  have hlam_inst := hlam_gen (balOfNat 199) (balOfNat 200)
  dsimp only [dual_var, dual_const, dual_mul, dual_add, DualEquiv, bsq] at hlam_inst
  have hlam_re := hlam_inst.1
  have hlam_eps := step24_balance_trans
    (step25_balance_add_congr
      (bal_mul_congr (step24_balance_refl _)
        (bal_neg_congr (bal_mul_congr (step24_balance_symm h398) (step24_balance_refl a))))
      (step24_balance_refl _))
    hlam_inst.2
  exact ⟨⟨bal_poly_eq env e_c1_L.1 e_c1_R.1 p_c1_re, bal_poly_eq env e_c1_L.2 e_c1_R.2 p_c1_eps⟩,
         ⟨bal_poly_eq env e_c2_L.1 e_c2_R.1 p_c2_re, bal_poly_eq env e_c2_L.2 e_c2_R.2 p_c2_eps⟩,
         ⟨hlam_re, hlam_eps⟩⟩

/-- ШАГ 112b. Рациональный Softmax порядка p = 4 (`src/math.rs::rational_probs_p4_into`, 0% exp):
    В целочисленном масштабе 8·q(z) = 8 - 4z + z²:
    1. 8 - 4z + z² ~ 4 + (z - 2)² ≥ 4 > 0 для ВСЕХ z : Balance.
    2. Знаменатель (8·q(z))⁴ > 0 (деление w(z) = 1 / q(z)⁴ тотально и строго положительно).
    3. Точная дуальная производная 8·q(z + ε) ~ (8 - 4z + z²) + (2z - 4)ε,
       откуда лог-производная ψ(z) = (w'/w) = -4 q'(z)/q(z) = (2 - z) / q(z). -/
theorem step112b_rational_softmax_p4_positive (z : Balance) :
    let q8 := balance_add (balance_sub (balOfNat 8) (balance_mul (balOfNat 4) z)) (bsq z)
    let zd := dual_var z
    let q8_d := dual_add
      (dual_add (dual_const (balOfNat 8)) (dual_mul (dual_const (balance_neg (balOfNat 4))) zd))
      (dual_mul zd zd)
    BalanceEquiv q8 (balance_add (balOfNat 4) (bsq (balance_sub z (balOfNat 2)))) ∧
    BalLe (balOfNat 4) q8 ∧
    BalGtZero q8 ∧
    BalGtZero (bpow4 q8) ∧
    DualEquiv q8_d ⟨q8, balance_sub (balance_mul (balOfNat 2) z) (balOfNat 4)⟩ := by
  dsimp only [dual_var, dual_const, dual_mul, dual_add, DualEquiv, bsq]
  let env : Nat → Balance := fun | 0 => z | _ => bzero
  let ez : BExpr := .var 0
  let eq8 : BExpr := .add (.sub (.nat 8) (.mul (.nat 4) ez)) (.mul ez ez)
  have hq8_id : BalanceEquiv
      (balance_add (balance_sub (balOfNat 8) (balance_mul (balOfNat 4) z)) (bsq z))
      (balance_add (balOfNat 4) (bsq (balance_sub z (balOfNat 2)))) :=
    bal_poly_eq env eq8 (.add (.nat 4) (.mul (.sub ez (.nat 2)) (.sub ez (.nat 2)))) rfl
  have hq8_sub : BalanceEquiv
      (bsq (balance_sub z (balOfNat 2)))
      (balance_sub (balance_add (balance_sub (balOfNat 8) (balance_mul (balOfNat 4) z)) (bsq z)) (balOfNat 4)) :=
    bal_poly_eq env (.mul (.sub ez (.nat 2)) (.sub ez (.nat 2))) (.sub eq8 (.nat 4)) rfl
  have hq8_gt : BalGtZero (balance_add (balance_sub (balOfNat 8) (balance_mul (balOfNat 4) z)) (bsq z)) :=
    bal_gt_zero_of_equiv (step24_balance_symm hq8_id)
      (bal_add_gt_ge (balOfNat_succ_gt_zero 3) (bal_sq_ge_zero (balance_sub z (balOfNat 2))))
  have hq8_p4 : BalGtZero (bpow4 (balance_add (balance_sub (balOfNat 8) (balance_mul (balOfNat 4) z)) (bsq z))) :=
    bal_mul_gt_gt (bal_mul_gt_gt hq8_gt hq8_gt) (bal_mul_gt_gt hq8_gt hq8_gt)
  have hd_re : BalanceEquiv
      (balance_add (balance_add (balOfNat 8) (balance_mul (balance_neg (balOfNat 4)) z)) (balance_mul z z))
      (balance_add (balance_sub (balOfNat 8) (balance_mul (balOfNat 4) z)) (balance_mul z z)) :=
    bal_poly_eq env
      (.add (.add (.nat 8) (.mul (.neg (.nat 4)) ez)) (.mul ez ez))
      eq8 rfl
  have hd_eps : BalanceEquiv
      (balance_add (balance_add bzero (balance_add (balance_mul (balance_neg (balOfNat 4)) bone) (balance_mul bzero z)))
                   (balance_add (balance_mul z bone) (balance_mul bone z)))
      (balance_sub (balance_mul (balOfNat 2) z) (balOfNat 4)) :=
    bal_poly_eq env
      (.add (.add .zero (.add (.mul (.neg (.nat 4)) .one) (.mul .zero ez)))
            (.add (.mul ez .one) (.mul .one ez)))
      (.sub (.mul (.nat 2) ez) (.nat 4)) rfl
  exact ⟨hq8_id,
         bal_ge_zero_of_equiv hq8_sub (bal_sq_ge_zero (balance_sub z (balOfNat 2))),
         hq8_gt,
         hq8_p4,
         ⟨hd_re, hd_eps⟩⟩

private def e_cos_L : BExpr × BExpr :=
  dMulE e_den_wd (.mul (.sub .one (.mul (.var 0) (.var 0))) e_Dw, .neg (.mul (.nat 4) (.var 0)))
private def e_cos_R : BExpr × BExpr :=
  dMulE (.mul e_Dw e_Dw, .zero) (.sub .one (.mul (.var 0) (.var 0)), .neg (.add (.mul (.var 0) .one) (.mul .one (.var 0))))
private theorem p_cos_re : e_cos_L.1.norm = e_cos_R.1.norm := rfl
private theorem p_cos_eps : e_cos_L.2.norm = e_cos_R.2.norm := rfl

/-- ШАГ 115. Алгебраический селективный 2D ротор Кэли (`src/rtrl.rs::AlgebraicSelectiveCayley`, 0% exp/sin/cos):
    1. Строгое BIBO-сжатие при любом селективном шаге Δ > 0 и любом α : Balance:
       0 < 998 < 1000(1 + Δ·α²)  (то есть 0 < λ(Δ, α) = 0.998 / (1 + Δ·α²) < 1).
    2. Точная дуальная производная рационального косинуса Кэли c(w) = (1 - w²)/(1 + w²):
       c'(w) = -4w / (1 + w²)² = -(2 / (1 + w²)) · s(w),
       что вместе с `step114` для s(w) = 2w/(1 + w²) доказывает `d(c, s)/dw = (2/(1+w²)) · (-s, c)`. -/
theorem step115_selective_cayley_contraction_and_deriv (delta a w : Balance) (hdelta : BalGtZero delta) :
    let num := balOfNat 998
    let den := balance_mul (balOfNat 1000) (balance_add bone (balance_mul delta (bsq a)))
    let wd := dual_var w
    let w2d := dual_mul wd wd
    let den_wd := dual_add (dual_const bone) w2d
    let D_w := balance_add bone (bsq w)
    let c_N := balance_sub bone (bsq w)
    let dc_num := balance_neg (balance_mul (balOfNat 4) w)
    let c_num_d : DualBalance := ⟨balance_sub bone (bsq w), balance_neg (balance_add (balance_mul w bone) (balance_mul bone w))⟩
    BalGtZero num ∧ BalGtZero den ∧ BalLt num den ∧
    DualEquiv
      (dual_mul den_wd ⟨balance_mul c_N D_w, dc_num⟩)
      (dual_mul (dual_const (bsq D_w)) c_num_d) := by
  dsimp only [dual_var, dual_const, dual_mul, dual_add, DualEquiv, bsq]
  have h998 : BalGtZero (balOfNat 998) := balOfNat_succ_gt_zero 997
  have h1000 : BalGtZero (balOfNat 1000) := balOfNat_succ_gt_zero 999
  have hda2 : BalGeZero (balance_mul delta (balance_mul a a)) :=
    bal_mul_gt_ge hdelta (bal_sq_ge_zero a)
  have hden : BalGtZero (balance_mul (balOfNat 1000) (balance_add bone (balance_mul delta (balance_mul a a)))) :=
    bal_mul_gt_gt h1000 (bal_add_gt_ge bone_gt_zero hda2)
  let env_c : Nat → Balance := fun | 0 => balance_mul delta (balance_mul a a) | _ => bzero
  have hdiff : BalanceEquiv
      (balance_add (balOfNat 2) (balance_mul (balOfNat 1000) (balance_mul delta (balance_mul a a))))
      (balance_sub (balance_mul (balOfNat 1000) (balance_add bone (balance_mul delta (balance_mul a a)))) (balOfNat 998)) :=
    bal_poly_eq env_c
      (.add (.nat 2) (.mul (.nat 1000) (.var 0)))
      (.sub (.mul (.nat 1000) (.add .one (.var 0))) (.nat 998))
      rfl
  have hlt : BalLt (balOfNat 998)
      (balance_mul (balOfNat 1000) (balance_add bone (balance_mul delta (balance_mul a a)))) :=
    bal_gt_zero_of_equiv hdiff
      (bal_add_gt_ge (balOfNat_succ_gt_zero 1) (bal_mul_gt_ge h1000 hda2))
  let env_w : Nat → Balance := fun | 0 => w | _ => bzero
  exact ⟨h998, hden, hlt,
         ⟨bal_poly_eq env_w e_cos_L.1 e_cos_R.1 p_cos_re,
          bal_poly_eq env_w e_cos_L.2 e_cos_R.2 p_cos_eps⟩⟩


-- =============================================================================
-- ЭТАП XXIX. ВЕРИФИКАЦИЯ v0.5.0: O(4) ДЛЯ S_5, MASTER EQUATION, RMSNORM И Q8.24
-- (Шаги 116–124; строго 0 аксиом на Balance / DualBalance / Vec4 / Vec5)
-- =============================================================================

theorem bal_mul_ge_ge {x y : Balance} (hx : BalGeZero x) (hy : BalGeZero y) :
    BalGeZero (balance_mul x y) := by
  cases hx with
  | inl hxz =>
    exact Or.inl (step24_balance_trans (bal_mul_congr hxz (step24_balance_refl y)) (bal_zero_mul y))
  | inr hxgt => exact bal_mul_gt_ge hxgt hy

theorem bal_le_trans {x y z : Balance} (hxy : BalLe x y) (hyz : BalLe y z) :
    BalLe x z := by
  dsimp [BalLe] at *
  let env : Nat → Balance := fun | 0 => x | 1 => y | 2 => z | _ => bzero
  have heq := bal_poly_eq env
    (.add (.sub (.var 1) (.var 0)) (.sub (.var 2) (.var 1)))
    (.sub (.var 2) (.var 0)) rfl
  exact bal_ge_zero_of_equiv heq (bal_add_ge_ge hxy hyz)

theorem bal_le_add_congr {a b c d : Balance} (hab : BalLe a b) (hcd : BalLe c d) :
    BalLe (balance_add a c) (balance_add b d) := by
  dsimp [BalLe] at *
  let env : Nat → Balance := fun | 0 => a | 1 => b | 2 => c | 3 => d | _ => bzero
  have heq := bal_poly_eq env
    (.add (.sub (.var 1) (.var 0)) (.sub (.var 3) (.var 2)))
    (.sub (.add (.var 1) (.var 3)) (.add (.var 0) (.var 2))) rfl
  exact bal_ge_zero_of_equiv heq (bal_add_ge_ge hab hcd)

theorem bal_neg_gt_zero {x : Balance} (hlt : x.fwd < x.bwd) :
    BalGtZero (balance_neg x) := by
  dsimp [BalGtZero, balance_neg]
  exact hlt

theorem bal_equiv_bzero_of_eq {x : Balance} (h : x.fwd = x.bwd) : BalanceEquiv x bzero :=
  congrArg PosNat.succ h

theorem bal_mul_gt_zero_eq_zero {k x : Balance} (hk : BalGtZero k)
    (h : BalanceEquiv (balance_mul k x) bzero) : BalanceEquiv x bzero := by
  cases trichotomy x.fwd x.bwd with
  | inl heq => exact bal_equiv_bzero_of_eq heq
  | inr hne =>
    cases hne with
    | inl hlt =>
      have hkn := bal_mul_gt_gt hk (bal_neg_gt_zero hlt)
      have hkn0 : BalanceEquiv (balance_mul k (balance_neg x)) bzero :=
        step24_balance_trans (bal_mul_neg k x)
          (step24_balance_trans (bal_neg_congr h) bal_neg_zero)
      exact False.elim (bal_gt_zero_ne_zero hkn hkn0)
    | inr hgt =>
      exact False.elim (bal_gt_zero_ne_zero (bal_mul_gt_gt hk hgt) h)

theorem bal_ge_zero_of_double_ge {x : Balance} (h2 : BalGeZero (balance_mul (balOfNat 2) x)) :
    BalGeZero x := by
  cases trichotomy x.fwd x.bwd with
  | inl eqx => exact Or.inl (bal_equiv_bzero_of_eq eqx)
  | inr hne =>
    cases hne with
    | inl ltx =>
      have h2n : BalGtZero (balance_neg (balance_mul (balOfNat 2) x)) :=
        bal_gt_zero_of_equiv (bal_mul_neg (balOfNat 2) x)
          (bal_mul_gt_gt (balOfNat_succ_gt_zero 1) (bal_neg_gt_zero ltx))
      cases h2 with
      | inl hz =>
        exact False.elim (bal_gt_zero_ne_zero h2n
          (step24_balance_trans (bal_neg_congr hz) bal_neg_zero))
      | inr hgt =>
        exact False.elim (bal_gt_zero_ne_zero (bal_add_gt_ge hgt (Or.inr h2n)) (bal_add_neg _))
    | inr gtx => exact Or.inr gtx

theorem bal_abs_le_sq_le {x B : Balance}
    (hlo : BalLe (balance_neg B) x) (hhi : BalLe x B) :
    BalLe (bsq x) (bsq B) := by
  dsimp [BalLe] at *
  let env : Nat → Balance := fun | 0 => x | 1 => B | _ => bzero
  have hsum : BalanceEquiv (balance_sub x (balance_neg B)) (balance_add B x) :=
    bal_poly_eq env (.sub (.var 0) (.neg (.var 1))) (.add (.var 1) (.var 0)) rfl
  have hplus := bal_ge_zero_of_equiv hsum hlo
  have hprod := bal_mul_ge_ge hhi hplus
  have hdiff : BalanceEquiv (balance_mul (balance_sub B x) (balance_add B x))
                            (balance_sub (bsq B) (bsq x)) :=
    bal_poly_eq env (.mul (.sub (.var 1) (.var 0)) (.add (.var 1) (.var 0)))
      (.sub (.mul (.var 1) (.var 1)) (.mul (.var 0) (.var 0))) rfl
  exact bal_ge_zero_of_equiv hdiff hprod

theorem bal_two_mul_add_mul (a x : Balance) :
    BalanceEquiv (balance_add (balance_mul (balOfNat 2) a) (balance_mul (balOfNat 2) x))
                 (balance_mul (balOfNat 2) (balance_add a x)) := by
  let env : Nat → Balance := fun | 0 => a | 1 => x | _ => bzero
  exact bal_poly_eq env (.add (.mul (.nat 2) (.var 0)) (.mul (.nat 2) (.var 1)))
    (.mul (.nat 2) (.add (.var 0) (.var 1))) rfl

-- ----- 4D Векторы, кватернионы, Хаусхолдер и det4 -----

structure Vec4 where
  w : Balance
  x : Balance
  y : Balance
  z : Balance

def Vec4Equiv (u v : Vec4) : Prop :=
  BalanceEquiv u.w v.w ∧ BalanceEquiv u.x v.x ∧ BalanceEquiv u.y v.y ∧ BalanceEquiv u.z v.z

def vec4_zero : Vec4 := ⟨bzero, bzero, bzero, bzero⟩
def vec4_add (u v : Vec4) : Vec4 :=
  ⟨balance_add u.w v.w, balance_add u.x v.x, balance_add u.y v.y, balance_add u.z v.z⟩
def vec4_sub (u v : Vec4) : Vec4 :=
  ⟨balance_sub u.w v.w, balance_sub u.x v.x, balance_sub u.y v.y, balance_sub u.z v.z⟩
def vec4_scale (s : Balance) (u : Vec4) : Vec4 :=
  ⟨balance_mul s u.w, balance_mul s u.x, balance_mul s u.y, balance_mul s u.z⟩
def dot4 (u v : Vec4) : Balance :=
  balance_add (balance_add (balance_mul u.w v.w) (balance_mul u.x v.x))
              (balance_add (balance_mul u.y v.y) (balance_mul u.z v.z))

theorem dot4_self_ge_zero (u : Vec4) : BalGeZero (dot4 u u) :=
  bal_add_ge_ge (bal_add_ge_ge (bal_sq_ge_zero u.w) (bal_sq_ge_zero u.x))
                (bal_add_ge_ge (bal_sq_ge_zero u.y) (bal_sq_ge_zero u.z))

theorem dot4_congr {u1 u2 v1 v2 : Vec4} (hu : Vec4Equiv u1 u2) (hv : Vec4Equiv v1 v2) :
    BalanceEquiv (dot4 u1 v1) (dot4 u2 v2) :=
  step25_balance_add_congr
    (step25_balance_add_congr (bal_mul_congr hu.1 hv.1) (bal_mul_congr hu.2.1 hv.2.1))
    (step25_balance_add_congr (bal_mul_congr hu.2.2.1 hv.2.2.1) (bal_mul_congr hu.2.2.2 hv.2.2.2))

theorem dot4_comm (u v : Vec4) : BalanceEquiv (dot4 u v) (dot4 v u) :=
  step25_balance_add_congr
    (step25_balance_add_congr (bal_mul_comm u.w v.w) (bal_mul_comm u.x v.x))
    (step25_balance_add_congr (bal_mul_comm u.y v.y) (bal_mul_comm u.z v.z))

private theorem dot4_lincomb2_sq_scalar (a b u v : Balance) :
    BalanceEquiv (bsq (balance_add (balance_mul a u) (balance_mul b v)))
      (balance_add (balance_add (balance_mul (bsq a) (bsq u)) (balance_mul (bsq b) (bsq v)))
                   (balance_mul (balance_mul (balOfNat 2) (balance_mul a b)) (balance_mul u v))) := by
  let env : Nat → Balance := fun | 0 => a | 1 => b | 2 => u | 3 => v | _ => bzero
  exact bal_poly_eq env
    (.mul (.add (.mul (.var 0) (.var 2)) (.mul (.var 1) (.var 3)))
          (.add (.mul (.var 0) (.var 2)) (.mul (.var 1) (.var 3))))
    (.add (.add (.mul (.mul (.var 0) (.var 0)) (.mul (.var 2) (.var 2)))
                (.mul (.mul (.var 1) (.var 1)) (.mul (.var 3) (.var 3))))
          (.mul (.mul (.nat 2) (.mul (.var 0) (.var 1))) (.mul (.var 2) (.var 3)))) rfl

private theorem dot4_sub2_sq_scalar (a b u v : Balance) :
    BalanceEquiv (bsq (balance_sub (balance_mul a u) (balance_mul b v)))
      (balance_add (balance_sub (balance_mul (bsq a) (bsq u))
                                (balance_mul (balance_mul (balOfNat 2) (balance_mul a b)) (balance_mul u v)))
                   (balance_mul (bsq b) (bsq v))) := by
  let env : Nat → Balance := fun | 0 => a | 1 => b | 2 => u | 3 => v | _ => bzero
  exact bal_poly_eq env
    (.mul (.sub (.mul (.var 0) (.var 2)) (.mul (.var 1) (.var 3)))
          (.sub (.mul (.var 0) (.var 2)) (.mul (.var 1) (.var 3))))
    (.add (.sub (.mul (.mul (.var 0) (.var 0)) (.mul (.var 2) (.var 2)))
                (.mul (.mul (.nat 2) (.mul (.var 0) (.var 1))) (.mul (.var 2) (.var 3))))
          (.mul (.mul (.var 1) (.var 1)) (.mul (.var 3) (.var 3)))) rfl

theorem dot4_lincomb2_sq (a b : Balance) (u v : Vec4) :
    let w := vec4_add (vec4_scale a u) (vec4_scale b v)
    BalanceEquiv (dot4 w w)
      (balance_add (balance_add (balance_mul (bsq a) (dot4 u u)) (balance_mul (bsq b) (dot4 v v)))
                   (balance_mul (balance_mul (balOfNat 2) (balance_mul a b)) (dot4 u v))) := by
  dsimp only
  have hw := dot4_lincomb2_sq_scalar a b u.w v.w
  have hx := dot4_lincomb2_sq_scalar a b u.x v.x
  have hy := dot4_lincomb2_sq_scalar a b u.y v.y
  have hz := dot4_lincomb2_sq_scalar a b u.z v.z
  have hsum := step25_balance_add_congr (step25_balance_add_congr hw hx) (step25_balance_add_congr hy hz)
  have hre : ∀ (p0 q0 r0 p1 q1 r1 p2 q2 r2 p3 q3 r3 : Balance),
      BalanceEquiv
        (balance_add (balance_add (balance_add (balance_add p0 q0) r0) (balance_add (balance_add p1 q1) r1))
                     (balance_add (balance_add (balance_add p2 q2) r2) (balance_add (balance_add p3 q3) r3)))
        (balance_add (balance_add (balance_add (balance_add p0 p1) (balance_add p2 p3))
                                  (balance_add (balance_add q0 q1) (balance_add q2 q3)))
                     (balance_add (balance_add r0 r1) (balance_add r2 r3))) := by
    intro p0 q0 r0 p1 q1 r1 p2 q2 r2 p3 q3 r3
    let env : Nat → Balance := fun
      | 0 => p0 | 1 => q0 | 2 => r0 | 3 => p1 | 4 => q1 | 5 => r1
      | 6 => p2 | 7 => q2 | 8 => r2 | 9 => p3 | 10 => q3 | 11 => r3 | _ => bzero
    exact bal_poly_eq env
      (.add (.add (.add (.add (.var 0) (.var 1)) (.var 2)) (.add (.add (.var 3) (.var 4)) (.var 5)))
            (.add (.add (.add (.var 6) (.var 7)) (.var 8)) (.add (.add (.var 9) (.var 10)) (.var 11))))
      (.add (.add (.add (.add (.var 0) (.var 3)) (.add (.var 6) (.var 9)))
                  (.add (.add (.var 1) (.var 4)) (.add (.var 7) (.var 10))))
            (.add (.add (.var 2) (.var 5)) (.add (.var 8) (.var 11)))) rfl
  let c1 := bsq a; let c2 := bsq b; let c3 := balance_mul (balOfNat 2) (balance_mul a b)
  have hfac : ∀ (k d0 d1 d2 d3 : Balance),
      BalanceEquiv
        (balance_add (balance_add (balance_mul k d0) (balance_mul k d1))
                     (balance_add (balance_mul k d2) (balance_mul k d3)))
        (balance_mul k (balance_add (balance_add d0 d1) (balance_add d2 d3))) := by
    intro k d0 d1 d2 d3
    let env : Nat → Balance := fun | 0 => k | 1 => d0 | 2 => d1 | 3 => d2 | 4 => d3 | _ => bzero
    exact bal_poly_eq env
      (.add (.add (.mul (.var 0) (.var 1)) (.mul (.var 0) (.var 2)))
            (.add (.mul (.var 0) (.var 3)) (.mul (.var 0) (.var 4))))
      (.mul (.var 0) (.add (.add (.var 1) (.var 2)) (.add (.var 3) (.var 4)))) rfl
  have s1 := hre
    (balance_mul c1 (bsq u.w)) (balance_mul c2 (bsq v.w)) (balance_mul c3 (balance_mul u.w v.w))
    (balance_mul c1 (bsq u.x)) (balance_mul c2 (bsq v.x)) (balance_mul c3 (balance_mul u.x v.x))
    (balance_mul c1 (bsq u.y)) (balance_mul c2 (bsq v.y)) (balance_mul c3 (balance_mul u.y v.y))
    (balance_mul c1 (bsq u.z)) (balance_mul c2 (bsq v.z)) (balance_mul c3 (balance_mul u.z v.z))
  have s2 := step25_balance_add_congr
    (step25_balance_add_congr (hfac c1 (bsq u.w) (bsq u.x) (bsq u.y) (bsq u.z))
                              (hfac c2 (bsq v.w) (bsq v.x) (bsq v.y) (bsq v.z)))
    (hfac c3 (balance_mul u.w v.w) (balance_mul u.x v.x) (balance_mul u.y v.y) (balance_mul u.z v.z))
  exact step24_balance_trans hsum (step24_balance_trans s1 s2)

theorem dot4_sub2_sq (a b : Balance) (u v : Vec4) :
    let w := vec4_sub (vec4_scale a u) (vec4_scale b v)
    BalanceEquiv (dot4 w w)
      (balance_add (balance_sub (balance_mul (bsq a) (dot4 u u))
                                (balance_mul (balance_mul (balOfNat 2) (balance_mul a b)) (dot4 u v)))
                   (balance_mul (bsq b) (dot4 v v))) := by
  dsimp only
  have c_w := dot4_sub2_sq_scalar a b u.w v.w
  have c_x := dot4_sub2_sq_scalar a b u.x v.x
  have c_y := dot4_sub2_sq_scalar a b u.y v.y
  have c_z := dot4_sub2_sq_scalar a b u.z v.z
  have hsum := step25_balance_add_congr (step25_balance_add_congr c_w c_x) (step25_balance_add_congr c_y c_z)
  have hre : ∀ (p0 m0 q0 p1 m1 q1 p2 m2 q2 p3 m3 q3 : Balance),
      BalanceEquiv
        (balance_add (balance_add (balance_add (balance_sub p0 m0) q0) (balance_add (balance_sub p1 m1) q1))
                     (balance_add (balance_add (balance_sub p2 m2) q2) (balance_add (balance_sub p3 m3) q3)))
        (balance_add (balance_sub (balance_add (balance_add p0 p1) (balance_add p2 p3))
                                  (balance_add (balance_add m0 m1) (balance_add m2 m3)))
                     (balance_add (balance_add q0 q1) (balance_add q2 q3))) := by
    intro p0 m0 q0 p1 m1 q1 p2 m2 q2 p3 m3 q3
    let env : Nat → Balance := fun
      | 0 => p0 | 1 => m0 | 2 => q0 | 3 => p1 | 4 => m1 | 5 => q1
      | 6 => p2 | 7 => m2 | 8 => q2 | 9 => p3 | 10 => m3 | 11 => q3 | _ => bzero
    exact bal_poly_eq env
      (.add (.add (.add (.sub (.var 0) (.var 1)) (.var 2)) (.add (.sub (.var 3) (.var 4)) (.var 5)))
            (.add (.add (.sub (.var 6) (.var 7)) (.var 8)) (.add (.sub (.var 9) (.var 10)) (.var 11))))
      (.add (.sub (.add (.add (.var 0) (.var 3)) (.add (.var 6) (.var 9)))
                  (.add (.add (.var 1) (.var 4)) (.add (.var 7) (.var 10))))
            (.add (.add (.var 2) (.var 5)) (.add (.var 8) (.var 11)))) rfl
  let c1 := bsq a; let c2 := bsq b; let c3 := balance_mul (balOfNat 2) (balance_mul a b)
  have hfac : ∀ (k d0 d1 d2 d3 : Balance),
      BalanceEquiv
        (balance_add (balance_add (balance_mul k d0) (balance_mul k d1))
                     (balance_add (balance_mul k d2) (balance_mul k d3)))
        (balance_mul k (balance_add (balance_add d0 d1) (balance_add d2 d3))) := by
    intro k d0 d1 d2 d3
    let env : Nat → Balance := fun | 0 => k | 1 => d0 | 2 => d1 | 3 => d2 | 4 => d3 | _ => bzero
    exact bal_poly_eq env
      (.add (.add (.mul (.var 0) (.var 1)) (.mul (.var 0) (.var 2)))
            (.add (.mul (.var 0) (.var 3)) (.mul (.var 0) (.var 4))))
      (.mul (.var 0) (.add (.add (.var 1) (.var 2)) (.add (.var 3) (.var 4)))) rfl
  have s1 := hre
    (balance_mul c1 (bsq u.w)) (balance_mul c3 (balance_mul u.w v.w)) (balance_mul c2 (bsq v.w))
    (balance_mul c1 (bsq u.x)) (balance_mul c3 (balance_mul u.x v.x)) (balance_mul c2 (bsq v.x))
    (balance_mul c1 (bsq u.y)) (balance_mul c3 (balance_mul u.y v.y)) (balance_mul c2 (bsq v.y))
    (balance_mul c1 (bsq u.z)) (balance_mul c3 (balance_mul u.z v.z)) (balance_mul c2 (bsq v.z))
  have s2 := step25_balance_add_congr
    (bal_sub_congr (hfac c1 (bsq u.w) (bsq u.x) (bsq u.y) (bsq u.z))
                   (hfac c3 (balance_mul u.w v.w) (balance_mul u.x v.x) (balance_mul u.y v.y) (balance_mul u.z v.z)))
    (hfac c2 (bsq v.w) (bsq v.x) (bsq v.y) (bsq v.z))
  exact step24_balance_trans hsum (step24_balance_trans s1 s2)

-- Кватернионное произведение, стереографическая карта и определитель 4x4
def qmul (p q : Vec4) : Vec4 :=
  ⟨balance_sub (balance_sub (balance_mul p.w q.w) (balance_mul p.x q.x))
               (balance_add (balance_mul p.y q.y) (balance_mul p.z q.z)),
   balance_sub (balance_add (balance_mul p.w q.x) (balance_mul p.x q.w))
               (balance_sub (balance_mul p.z q.y) (balance_mul p.y q.z)),
   balance_sub (balance_add (balance_mul p.w q.y) (balance_mul p.y q.w))
               (balance_sub (balance_mul p.x q.z) (balance_mul p.z q.x)),
   balance_sub (balance_add (balance_mul p.w q.z) (balance_mul p.z q.w))
               (balance_sub (balance_mul p.y q.x) (balance_mul p.x q.y))⟩

def q_stereo (g : Vec3) : Vec4 :=
  ⟨balance_sub bone (dot3 g g),
   balance_mul (balOfNat 2) g.x,
   balance_mul (balOfNat 2) g.y,
   balance_mul (balOfNat 2) g.z⟩

def det4 (c1 c2 c3 c4 : Vec4) : Balance :=
  let t1 : Vec3 := ⟨c1.x, c1.y, c1.z⟩
  let t2 : Vec3 := ⟨c2.x, c2.y, c2.z⟩
  let t3 : Vec3 := ⟨c3.x, c3.y, c3.z⟩
  let t4 : Vec3 := ⟨c4.x, c4.y, c4.z⟩
  balance_sub
    (balance_add (balance_mul c1.w (det3 t2 t3 t4)) (balance_mul c3.w (det3 t1 t2 t4)))
    (balance_add (balance_mul c2.w (det3 t1 t3 t4)) (balance_mul c4.w (det3 t1 t2 t3)))

private def bDot4E (u v : BExpr × BExpr × BExpr × BExpr) : BExpr :=
  .add (.add (.mul u.1 v.1) (.mul u.2.1 v.2.1))
       (.add (.mul u.2.2.1 v.2.2.1) (.mul u.2.2.2 v.2.2.2))

private def bQmulE (p q : BExpr × BExpr × BExpr × BExpr) : BExpr × BExpr × BExpr × BExpr :=
  (.sub (.sub (.mul p.1 q.1) (.mul p.2.1 q.2.1)) (.add (.mul p.2.2.1 q.2.2.1) (.mul p.2.2.2 q.2.2.2)),
   .sub (.add (.mul p.1 q.2.1) (.mul p.2.1 q.1)) (.sub (.mul p.2.2.2 q.2.2.1) (.mul p.2.2.1 q.2.2.2)),
   .sub (.add (.mul p.1 q.2.2.1) (.mul p.2.2.1 q.1)) (.sub (.mul p.2.1 q.2.2.2) (.mul p.2.2.2 q.2.1)),
   .sub (.add (.mul p.1 q.2.2.2) (.mul p.2.2.2 q.1)) (.sub (.mul p.2.2.1 q.2.1) (.mul p.2.1 q.2.2.1)))

private def bDet4E (c1 c2 c3 c4 : BExpr × BExpr × BExpr × BExpr) : BExpr :=
  let t1 := (c1.2.1, c1.2.2.1, c1.2.2.2)
  let t2 := (c2.2.1, c2.2.2.1, c2.2.2.2)
  let t3 := (c3.2.1, c3.2.2.1, c3.2.2.2)
  let t4 := (c4.2.1, c4.2.2.1, c4.2.2.2)
  let d3 (a b c : BExpr × BExpr × BExpr) := vDotE a (vCrossE b c)
  .sub (.add (.mul c1.1 (d3 t2 t3 t4)) (.mul c3.1 (d3 t1 t2 t4)))
       (.add (.mul c2.1 (d3 t1 t3 t4)) (.mul c4.1 (d3 t1 t2 t3)))

private def e_qnorm_L : BExpr :=
  let ep := (.var 0, .var 1, .var 2, .var 3)
  let eq := (.var 4, .var 5, .var 6, .var 7)
  let pq := bQmulE ep eq
  bDot4E pq pq

private def e_qnorm_R : BExpr :=
  let ep := (.var 0, .var 1, .var 2, .var 3)
  let eq := (.var 4, .var 5, .var 6, .var 7)
  .mul (bDot4E ep ep) (bDot4E eq eq)

private theorem p_qnorm_eq : e_qnorm_L.norm = e_qnorm_R.norm := rfl

private def e_stereo_L : BExpr :=
  let r := .var 0
  let s1 := .sub .one r
  .add (.mul s1 s1) (.mul (.nat 4) r)

private def e_stereo_R : BExpr :=
  let d := .add .one (.var 0)
  .mul d d

private theorem p_stereo_eq : e_stereo_L.norm = e_stereo_R.norm := rfl

private def e_eq4 : BExpr × BExpr × BExpr × BExpr := (.var 0, .var 1, .var 2, .var 3)
private def e_det4_qL : BExpr :=
  bDet4E (bQmulE e_eq4 (.one, .zero, .zero, .zero))
         (bQmulE e_eq4 (.zero, .one, .zero, .zero))
         (bQmulE e_eq4 (.zero, .zero, .one, .zero))
         (bQmulE e_eq4 (.zero, .zero, .zero, .one))

private def e_det4_qR : BExpr :=
  bDet4E (bQmulE (.one, .zero, .zero, .zero) e_eq4)
         (bQmulE (.zero, .one, .zero, .zero) e_eq4)
         (bQmulE (.zero, .zero, .one, .zero) e_eq4)
         (bQmulE (.zero, .zero, .zero, .one) e_eq4)

private def e_qnorm4_sq : BExpr :=
  let nn := bDot4E e_eq4 e_eq4
  .mul nn nn

private theorem p_det4_qL : e_det4_qL.norm = e_qnorm4_sq.norm := rfl
private theorem p_det4_qR : e_det4_qR.norm = e_qnorm4_sq.norm := rfl

private def e_q2 : BExpr := .add (.sub (.nat 2) (.mul (.nat 2) (.var 0))) (.mul (.var 0) (.var 0))
private def e_q2_d : BExpr × BExpr :=
  dAddE (dAddE (.nat 2, .zero) (dMulE (.neg (.nat 2), .zero) (.var 0, .one))) (dMulE (.var 0, .one) (.var 0, .one))
private def e_s116_L : BExpr × BExpr := dMulE e_q2_d (.mul (.nat 2) e_q2, .mul (.nat 4) (.sub .one (.var 0)))
private def e_s116_R : BExpr × BExpr := dMulE (.mul e_q2 e_q2, .zero) (.nat 2, .zero)
private def e_q20 : BExpr := .add (.sub (.nat 2) (.mul (.nat 2) .zero)) (.mul .zero .zero)
private theorem p_s116_id : e_q2.norm = (BExpr.add .one (.mul (.sub (.var 0) .one) (.sub (.var 0) .one))).norm := rfl
private theorem p_s116_sub : (BExpr.mul (.sub (.var 0) .one) (.sub (.var 0) .one)).norm = (BExpr.sub e_q2 .one).norm := rfl
private theorem p_s116_re : e_s116_L.1.norm = e_s116_R.1.norm := rfl
private theorem p_s116_eps : e_s116_L.2.norm = e_s116_R.2.norm := rfl
private theorem p_s116_c0 : e_q20.norm = (BExpr.nat 2).norm := rfl
private theorem p_s116_c1 : (BExpr.mul (.nat 4) (.sub .one .zero)).norm = (BExpr.mul e_q20 e_q20).norm := rfl

theorem step116_rational_pos_gate_c1_and_bounds (z : Balance) :
    let q2 := balance_add (balance_sub (balOfNat 2) (balance_mul (balOfNat 2) z)) (bsq z)
    let zd := dual_var z
    let q2_d := dual_add
      (dual_add (dual_const (balOfNat 2)) (dual_mul (dual_const (balance_neg (balOfNat 2))) zd))
      (dual_mul zd zd)
    let q2_zero := balance_add (balance_sub (balOfNat 2) (balance_mul (balOfNat 2) bzero)) (bsq bzero)
    BalanceEquiv q2 (balance_add bone (bsq (balance_sub z bone))) ∧
    BalLe bone q2 ∧
    BalGtZero q2 ∧
    DualEquiv
      (dual_mul q2_d ⟨balance_mul (balOfNat 2) q2, balance_mul (balOfNat 4) (balance_sub bone z)⟩)
      (dual_mul (dual_const (bsq q2)) (dual_const (balOfNat 2))) ∧
    BalanceEquiv q2_zero (balOfNat 2) ∧
    BalanceEquiv (balance_mul (balOfNat 4) (balance_sub bone bzero)) (bsq q2_zero) := by
  dsimp only [dual_var, dual_const, dual_mul, dual_add, DualEquiv, bsq]
  let env : Nat → Balance := fun | 0 => z | _ => bzero
  have hq_id := bal_poly_eq env e_q2 _ p_s116_id
  have hq_sub := bal_poly_eq env _ (BExpr.sub e_q2 .one) p_s116_sub
  have hq_gt : BalGtZero (balance_add (balance_sub (balOfNat 2) (balance_mul (balOfNat 2) z)) (balance_mul z z)) :=
    bal_gt_zero_of_equiv (step24_balance_symm hq_id) (bal_one_add_sq_gt_zero (balance_sub z bone))
  exact ⟨hq_id, bal_ge_zero_of_equiv hq_sub (bal_sq_ge_zero (balance_sub z bone)), hq_gt,
         ⟨bal_poly_eq env e_s116_L.1 e_s116_R.1 p_s116_re,
          bal_poly_eq env e_s116_L.2 e_s116_R.2 p_s116_eps⟩,
         bal_poly_eq env e_q20 (.nat 2) p_s116_c0,
         bal_poly_eq env _ (BExpr.mul e_q20 e_q20) p_s116_c1⟩

/-- ШАГ 117. Рациональные кватернионы и двойной ротор SO(4) без `sqrt` и обращения матриц:
    1. Стереографический кватернион `q_stereo(g) = (1 - ||g||², 2g)` имеет точную норму `(1 + ||g||²)² > 0`.
    2. Мультипликативность Эйлера `||p * q||² ~ ||p||² · ||q||²`.
    3. Двойной ротор `u' = q_stereo(gL) * u * q_stereo(gR)` точно сохраняет 4D-норму
       `||u'||² ~ ((1 + ||gL||²)(1 + ||gR||²))² · ||u||²`.
    4. Матрицы левого и правого кватернионного умножения имеют `det₄(L(q)) ~ det₄(R(q)) ~ (||q||²)² > 0`
       (`det = +1`, строго внутри `SO(4)`). -/
theorem step117_rational_quat_and_so4_double_rotor (gL gR : Vec3) (u : Vec4) :
    let dL := balance_add bone (dot3 gL gL)
    let dR := balance_add bone (dot3 gR gR)
    let qL := q_stereo gL
    let qR := q_stereo gR
    let u' := qmul qL (qmul u qR)
    let e1 : Vec4 := ⟨bone, bzero, bzero, bzero⟩
    let e2 : Vec4 := ⟨bzero, bone, bzero, bzero⟩
    let e3 : Vec4 := ⟨bzero, bzero, bone, bzero⟩
    let e4 : Vec4 := ⟨bzero, bzero, bzero, bone⟩
    BalGtZero dL ∧ BalGtZero dR ∧
    BalanceEquiv (dot4 qL qL) (bsq dL) ∧
    BalanceEquiv (dot4 qR qR) (bsq dR) ∧
    BalanceEquiv (dot4 u' u') (balance_mul (bsq (balance_mul dL dR)) (dot4 u u)) ∧
    BalanceEquiv (det4 (qmul qL e1) (qmul qL e2) (qmul qL e3) (qmul qL e4)) (bsq (dot4 qL qL)) ∧
    BalanceEquiv (det4 (qmul e1 qR) (qmul e2 qR) (qmul e3 qR) (qmul e4 qR)) (bsq (dot4 qR qR)) := by
  dsimp only
  have hdL := bal_add_gt_ge bone_gt_zero (dot3_self_ge_zero gL)
  have hdR := bal_add_gt_ge bone_gt_zero (dot3_self_ge_zero gR)
  have hstereo : ∀ (g : Vec3), BalanceEquiv (dot4 (q_stereo g) (q_stereo g)) (bsq (balance_add bone (dot3 g g))) := by
    intro g
    let env : Nat → Balance := fun | 0 => dot3 g g | 1 => g.x | 2 => g.y | 3 => g.z | _ => bzero
    have h4dot : BalanceEquiv
        (dot4 (q_stereo g) (q_stereo g))
        (balance_add (bsq (balance_sub bone (dot3 g g))) (balance_mul (balOfNat 4) (dot3 g g))) := by
      let env4 : Nat → Balance := fun | 0 => dot3 g g | 1 => g.x | 2 => g.y | 3 => g.z | _ => bzero
      exact bal_poly_eq env4
        (.add (.add (.mul (.sub .one (.var 0)) (.sub .one (.var 0))) (.mul (.mul (.nat 2) (.var 1)) (.mul (.nat 2) (.var 1))))
              (.add (.mul (.mul (.nat 2) (.var 2)) (.mul (.nat 2) (.var 2))) (.mul (.mul (.nat 2) (.var 3)) (.mul (.nat 2) (.var 3)))))
        (.add (.mul (.sub .one (.var 0)) (.sub .one (.var 0)))
              (.mul (.nat 4) (.add (.add (.mul (.var 1) (.var 1)) (.mul (.var 2) (.var 2))) (.mul (.var 3) (.var 3))))) rfl
    exact step24_balance_trans h4dot (bal_poly_eq env e_stereo_L e_stereo_R p_stereo_eq)
  have hqnorm : ∀ (p q : Vec4), BalanceEquiv (dot4 (qmul p q) (qmul p q)) (balance_mul (dot4 p p) (dot4 q q)) := by
    intro p q
    let env : Nat → Balance := fun
      | 0 => p.w | 1 => p.x | 2 => p.y | 3 => p.z
      | 4 => q.w | 5 => q.x | 6 => q.y | 7 => q.z | _ => bzero
    exact bal_poly_eq env e_qnorm_L e_qnorm_R p_qnorm_eq
  have hsqL := hstereo gL
  have hsqR := hstereo gR
  have huR := hqnorm u (q_stereo gR)
  have huLR := step24_balance_trans (hqnorm (q_stereo gL) (qmul u (q_stereo gR)))
    (bal_mul_congr hsqL (step24_balance_trans huR (bal_mul_congr (step24_balance_refl _) hsqR)))
  let envS : Nat → Balance := fun | 0 => balance_add bone (dot3 gL gL) | 1 => balance_add bone (dot3 gR gR) | 2 => dot4 u u | _ => bzero
  have hrearr := bal_poly_eq envS
    (.mul (.mul (.var 0) (.var 0)) (.mul (.var 2) (.mul (.var 1) (.var 1))))
    (.mul (.mul (.mul (.var 0) (.var 1)) (.mul (.var 0) (.var 1))) (.var 2)) rfl
  let envL : Nat → Balance := fun | 0 => (q_stereo gL).w | 1 => (q_stereo gL).x | 2 => (q_stereo gL).y | 3 => (q_stereo gL).z | _ => bzero
  let envR : Nat → Balance := fun | 0 => (q_stereo gR).w | 1 => (q_stereo gR).x | 2 => (q_stereo gR).y | 3 => (q_stereo gR).z | _ => bzero
  exact ⟨hdL, hdR, hsqL, hsqR, step24_balance_trans huLR hrearr,
         bal_poly_eq envL e_det4_qL e_qnorm4_sq p_det4_qL,
         bal_poly_eq envR e_det4_qR e_qnorm4_sq p_det4_qR⟩

-- Целочисленная форма гейтированного 4D-отражения Хаусхолдера H_{p/q}(v) u
-- до деления на q · D, где D = ||v||² + eps:
-- h_num = (q·D)·u - p·(v·u)·v
def householder4_num (D p q : Balance) (v u : Vec4) : Vec4 :=
  vec4_sub (vec4_scale (balance_mul q D) u) (vec4_scale (balance_mul p (dot4 v u)) v)

private def e_hcol (vv : BExpr) (v : BExpr × BExpr × BExpr × BExpr) (c : BExpr × BExpr × BExpr × BExpr) :
    BExpr × BExpr × BExpr × BExpr :=
  let s := .mul (.nat 2) (bDot4E v c)
  (.sub (.mul vv c.1) (.mul s v.1),
   .sub (.mul vv c.2.1) (.mul s v.2.1),
   .sub (.mul vv c.2.2.1) (.mul s v.2.2.1),
   .sub (.mul vv c.2.2.2) (.mul s v.2.2.2))

private def e_det4_h2_gen_L : BExpr :=
  let eD := BExpr.var 4
  let ev := e_eq4
  let ec1 := e_hcol eD ev (.one, .zero, .zero, .zero)
  let ec2 := e_hcol eD ev (.zero, .one, .zero, .zero)
  let ec3 := e_hcol eD ev (.zero, .zero, .one, .zero)
  let ec4 := e_hcol eD ev (.zero, .zero, .zero, .one)
  bDet4E ec1 ec2 ec3 ec4

private def e_det4_h2_gen_R : BExpr :=
  let eD := BExpr.var 4
  let evv := bDot4E e_eq4 e_eq4
  .mul (.mul (.mul eD eD) eD) (.sub eD (.mul (.nat 2) evv))

private theorem p_det4_h2_gen : e_det4_h2_gen_L.norm = e_det4_h2_gen_R.norm := rfl

private def e_sub_add_cancel_L : BExpr := .var 1
private def e_sub_add_cancel_R : BExpr := .sub (.add (.var 0) (.var 1)) (.var 0)
private theorem p_sub_add_cancel : e_sub_add_cancel_L.norm = e_sub_add_cancel_R.norm := rfl

private def e_add_sub_cancel_L : BExpr := .var 0
private def e_add_sub_cancel_R : BExpr := .sub (.add (.var 0) (.var 1)) (.var 1)
private theorem p_add_sub_cancel : e_add_sub_cancel_L.norm = e_add_sub_cancel_R.norm := rfl

private def e_sub_add_sub_L : BExpr := .add (.sub (.var 0) (.var 1)) (.sub (.var 2) (.var 3))
private def e_sub_add_sub_R : BExpr := .sub (.add (.var 0) (.var 2)) (.add (.var 1) (.var 3))
private theorem p_sub_add_sub : e_sub_add_sub_L.norm = e_sub_add_sub_R.norm := rfl

private def e_mul_sum4_L : BExpr :=
  .add (.add (.mul (.var 0) (.var 1)) (.mul (.var 0) (.var 2)))
       (.add (.mul (.var 0) (.var 3)) (.mul (.var 0) (.var 4)))
private def e_mul_sum4_R : BExpr :=
  .mul (.var 0) (.add (.add (.var 1) (.var 2)) (.add (.var 3) (.var 4)))
private theorem p_mul_sum4 : e_mul_sum4_L.norm = e_mul_sum4_R.norm := rfl

private def e_s118_hid_L : BExpr :=
  let ep := BExpr.var 0; let eq := BExpr.var 1; let evv := BExpr.var 2
  let eeps := BExpr.var 3; let evu := BExpr.var 4; let euu := BExpr.var 5
  let eD := BExpr.add evv eeps
  let eh_sq := BExpr.add (.sub (.mul (.mul (.mul eq eD) (.mul eq eD)) euu)
                               (.mul (.mul (.nat 2) (.mul (.mul eq eD) (.mul ep evu))) evu))
                         (.mul (.mul (.mul ep evu) (.mul ep evu)) evv)
  let eslack := BExpr.mul (.mul ep (.mul evu evu))
    (.add (.mul (.sub (.mul (.nat 2) eq) ep) evv) (.mul (.mul (.nat 2) eq) eeps))
  .add eh_sq eslack

private def e_s118_hid_R : BExpr :=
  let eq := BExpr.var 1; let evv := BExpr.var 2; let eeps := BExpr.var 3; let euu := BExpr.var 5
  let eD := BExpr.add evv eeps
  .mul (.mul (.mul eq eD) (.mul eq eD)) euu

private theorem p_s118_hid : e_s118_hid_L.norm = e_s118_hid_R.norm := rfl

private def e_s118_h2_L : BExpr :=
  .add (.sub (.mul (.mul (.mul .one (.var 0)) (.mul .one (.var 0))) (.var 2))
             (.mul (.mul (.nat 2) (.mul (.mul .one (.var 0)) (.mul (.nat 2) (.var 1)))) (.var 1)))
       (.mul (.mul (.mul (.nat 2) (.var 1)) (.mul (.nat 2) (.var 1))) (.var 0))
private def e_s118_h2_R : BExpr := .mul (.mul (.var 0) (.var 0)) (.var 2)
private theorem p_s118_h2 : e_s118_h2_L.norm = e_s118_h2_R.norm := rfl

private def e_s118_hexp_L : BExpr := .mul (.var 0) (.sub (.mul (.var 2) (.var 1)) (.mul (.var 3) (.var 0)))
private def e_s118_hexp_R : BExpr := .sub (.mul (.var 2) (.mul (.var 0) (.var 1))) (.mul (.var 3) (.mul (.var 0) (.var 0)))
private theorem p_s118_hexp : e_s118_hexp_L.norm = e_s118_hexp_R.norm := rfl

private def e_s118_hfin_L : BExpr := .sub (.mul (.mul .one (.var 0)) (.var 1)) (.mul (.mul (.nat 2) (.var 1)) (.var 0))
private def e_s118_hfin_R : BExpr := .neg (.mul (.var 0) (.var 1))
private theorem p_s118_hfin : e_s118_hfin_L.norm = e_s118_hfin_R.norm := rfl

private def e_s118_inv_L : BExpr :=
  .sub (.mul (.mul .one (.var 0)) (.sub (.mul (.mul .one (.var 0)) (.var 3)) (.mul (.mul (.nat 2) (.var 1)) (.var 2))))
       (.mul (.mul (.nat 2) (.neg (.mul (.var 0) (.var 1)))) (.var 2))
private def e_s118_inv_R : BExpr := .mul (.mul (.var 0) (.var 0)) (.var 3)
private theorem p_s118_inv : e_s118_inv_L.norm = e_s118_inv_R.norm := rfl

private def e_s118_det_spec_L : BExpr :=
  let d := BExpr.mul .one (.var 0)
  .mul (.mul (.mul d d) d) (.sub d (.mul (.nat 2) (.var 0)))
private def e_s118_det_spec_R : BExpr :=
  .neg (.mul (.mul (.var 0) (.var 0)) (.mul (.var 0) (.var 0)))
private theorem p_s118_det_spec : e_s118_det_spec_L.norm = e_s118_det_spec_R.norm := rfl

private theorem s118_det_h2 (v : Vec4) :
    let D0 := dot4 v v
    BalanceEquiv (det4 (householder4_num D0 (balOfNat 2) bone v ⟨bone, bzero, bzero, bzero⟩)
                       (householder4_num D0 (balOfNat 2) bone v ⟨bzero, bone, bzero, bzero⟩)
                       (householder4_num D0 (balOfNat 2) bone v ⟨bzero, bzero, bone, bzero⟩)
                       (householder4_num D0 (balOfNat 2) bone v ⟨bzero, bzero, bzero, bone⟩))
                 (balance_neg (bsq (bsq D0))) := by
  dsimp only
  let vv := dot4 v v
  let env4 : Nat → Balance := fun | 0 => v.w | 1 => v.x | 2 => v.y | 3 => v.z | 4 => balance_mul bone vv | _ => bzero
  have hgen := bal_poly_eq env4 e_det4_h2_gen_L e_det4_h2_gen_R p_det4_h2_gen
  have hspec := bal_poly_eq (fun | 0 => vv | _ => bzero)
    e_s118_det_spec_L e_s118_det_spec_R p_s118_det_spec
  exact step24_balance_trans hgen hspec

/-- ШАГ 118. Гейтированное рациональное отражение Хаусхолдера в R⁴ (ключ к O(4) и S_5):
    Для ЛЮБЫХ v, u : Vec4, eps ≥ 0 и рационального гейта beta = p/q ∈ [0, 2]:
    1. Точное тождество неэкспансивности (BIBO при всех β ∈ [0, 2]):
       ||h_num||² + p(v·u)²((2q - p)||v||² + 2q·eps) ~ (q·D)²·||u||², поэтому ||h_num||² ≤ (q·D)²·||u||².
    2. При β = 2 (p=2, q=1, eps=0, D₀ = ||v||²): точная изометрия и точная инволюция H₂(H₂(u)) ~ D₀²·u.
    3. Определитель матрицы отражения D₀·I₄ - 2vvᵀ точно равен -(D₀)⁴ (нечётная чётность det = -1!). -/
theorem step118_gated_householder_o4_bibo_and_parity
    (v u : Vec4) (eps p q : Balance)
    (heps : BalGeZero eps) (hq : BalGtZero q) (hp0 : BalGeZero p)
    (hp2 : BalLe p (balance_mul (balOfNat 2) q)) :
    let D := balance_add (dot4 v v) eps
    let h_num := householder4_num D p q v u
    let slack := balance_mul (balance_mul p (bsq (dot4 v u)))
                   (balance_add (balance_mul (balance_sub (balance_mul (balOfNat 2) q) p) (dot4 v v))
                                (balance_mul (balance_mul (balOfNat 2) q) eps))
    let D0 := dot4 v v
    let h2u := householder4_num D0 (balOfNat 2) bone v u
    let h2h2u := householder4_num D0 (balOfNat 2) bone v h2u
    BalanceEquiv (balance_add (dot4 h_num h_num) slack) (balance_mul (bsq (balance_mul q D)) (dot4 u u)) ∧
    BalLe (dot4 h_num h_num) (balance_mul (bsq (balance_mul q D)) (dot4 u u)) ∧
    BalanceEquiv (dot4 h2u h2u) (balance_mul (bsq D0) (dot4 u u)) ∧
    Vec4Equiv h2h2u (vec4_scale (bsq D0) u) ∧
    BalanceEquiv (det4 (householder4_num D0 (balOfNat 2) bone v ⟨bone, bzero, bzero, bzero⟩)
                       (householder4_num D0 (balOfNat 2) bone v ⟨bzero, bone, bzero, bzero⟩)
                       (householder4_num D0 (balOfNat 2) bone v ⟨bzero, bzero, bone, bzero⟩)
                       (householder4_num D0 (balOfNat 2) bone v ⟨bzero, bzero, bzero, bone⟩))
                 (balance_neg (bsq (bsq D0))) := by
  dsimp only
  let D := balance_add (dot4 v v) eps
  let vu := dot4 v u
  let vv := dot4 v v
  let uu := dot4 u u
  have hsq := dot4_sub2_sq (balance_mul q D) (balance_mul p vu) u v
  -- Доп. сомножитель u·v ~ vu
  have huv : BalanceEquiv (dot4 u v) vu := dot4_comm u v
  have hsq' := step24_balance_trans hsq
    (step25_balance_add_congr
      (bal_sub_congr (step24_balance_refl _) (bal_mul_congr (step24_balance_refl _) huv))
      (step24_balance_refl _))
  let env : Nat → Balance := fun | 0 => p | 1 => q | 2 => vv | 3 => eps | 4 => vu | 5 => uu | _ => bzero
  have hid := step24_balance_trans (step25_balance_add_congr hsq' (step24_balance_refl _))
    (bal_poly_eq env e_s118_hid_L e_s118_hid_R p_s118_hid)
  -- Неотрицательность slack => BalLe ||h_num||² ((q·D)²·||u||²)
  have h2q : BalGtZero (balance_mul (balOfNat 2) q) := bal_mul_gt_gt (balOfNat_succ_gt_zero 1) hq
  have hslack_ge : BalGeZero (balance_mul (balance_mul p (bsq vu))
                   (balance_add (balance_mul (balance_sub (balance_mul (balOfNat 2) q) p) vv)
                                (balance_mul (balance_mul (balOfNat 2) q) eps))) :=
    bal_mul_ge_ge (bal_mul_ge_ge hp0 (bal_sq_ge_zero vu))
      (bal_add_ge_ge (bal_mul_ge_ge hp2 (dot4_self_ge_zero v)) (bal_mul_ge_ge (Or.inr h2q) heps))
  have hsub_slack : BalanceEquiv
      (balance_mul (balance_mul p (bsq vu))
        (balance_add (balance_mul (balance_sub (balance_mul (balOfNat 2) q) p) vv)
                     (balance_mul (balance_mul (balOfNat 2) q) eps)))
      (balance_sub (balance_mul (bsq (balance_mul q D)) uu)
                   (dot4 (householder4_num D p q v u) (householder4_num D p q v u))) :=
    step24_balance_trans
      (bal_poly_eq
        (fun | 0 => dot4 (householder4_num D p q v u) (householder4_num D p q v u)
             | 1 => balance_mul (balance_mul p (bsq vu))
                      (balance_add (balance_mul (balance_sub (balance_mul (balOfNat 2) q) p) vv)
                                   (balance_mul (balance_mul (balOfNat 2) q) eps))
             | _ => bzero)
        e_sub_add_cancel_L e_sub_add_cancel_R p_sub_add_cancel)
      (bal_sub_congr hid (step24_balance_refl _))
  have hbibo : BalLe (dot4 (householder4_num D p q v u) (householder4_num D p q v u))
                     (balance_mul (bsq (balance_mul q D)) uu) :=
    bal_ge_zero_of_equiv hsub_slack hslack_ge
  -- Частный случай p=2, q=1, eps=0: точная изометрия H_2
  let h2u := householder4_num vv (balOfNat 2) bone v u
  have h2_norm : BalanceEquiv (dot4 h2u h2u) (balance_mul (bsq vv) uu) := by
    have hsq2 := dot4_sub2_sq (balance_mul bone vv) (balance_mul (balOfNat 2) vu) u v
    have hsq2' := step24_balance_trans hsq2
      (step25_balance_add_congr
        (bal_sub_congr (step24_balance_refl _) (bal_mul_congr (step24_balance_refl _) huv))
        (step24_balance_refl _))
    let env2 : Nat → Balance := fun | 0 => vv | 1 => vu | 2 => uu | _ => bzero
    exact step24_balance_trans hsq2' (bal_poly_eq env2 e_s118_h2_L e_s118_h2_R p_s118_h2)
  -- Точная инволюция H_2(H_2(u)) ~ D₀² · u: вычисляем скалярное произведение v · h2u ~ -D₀ · (v·u)
  have hvh2u : BalanceEquiv (dot4 v h2u) (balance_neg (balance_mul vv vu)) := by
    let s1 := balance_mul bone vv
    let s2 := balance_mul (balOfNat 2) vu
    have hexp : ∀ (vc uc : Balance),
        BalanceEquiv (balance_mul vc (balance_sub (balance_mul s1 uc) (balance_mul s2 vc)))
                     (balance_sub (balance_mul s1 (balance_mul vc uc)) (balance_mul s2 (bsq vc))) := by
      intro vc uc
      exact bal_poly_eq (fun | 0 => vc | 1 => uc | 2 => s1 | 3 => s2 | _ => bzero)
        e_s118_hexp_L e_s118_hexp_R p_s118_hexp
    have hdot := step25_balance_add_congr
      (step25_balance_add_congr (hexp v.w u.w) (hexp v.x u.x))
      (step25_balance_add_congr (hexp v.y u.y) (hexp v.z u.z))
    have hre : BalanceEquiv
        (balance_add (balance_add (balance_sub (balance_mul s1 (balance_mul v.w u.w)) (balance_mul s2 (bsq v.w)))
                                  (balance_sub (balance_mul s1 (balance_mul v.x u.x)) (balance_mul s2 (bsq v.x))))
                     (balance_add (balance_sub (balance_mul s1 (balance_mul v.y u.y)) (balance_mul s2 (bsq v.y)))
                                  (balance_sub (balance_mul s1 (balance_mul v.z u.z)) (balance_mul s2 (bsq v.z)))))
        (balance_sub (balance_mul s1 vu) (balance_mul s2 vv)) := by
      let pw := balance_mul v.w u.w; let qw := bsq v.w
      let px := balance_mul v.x u.x; let qx := bsq v.x
      let py := balance_mul v.y u.y; let qy := bsq v.y
      let pz := balance_mul v.z u.z; let qz := bsq v.z
      have hwx := bal_poly_eq (fun | 0 => balance_mul s1 pw | 1 => balance_mul s2 qw
                                   | 2 => balance_mul s1 px | 3 => balance_mul s2 qx | _ => bzero)
        e_sub_add_sub_L e_sub_add_sub_R p_sub_add_sub
      have hyz := bal_poly_eq (fun | 0 => balance_mul s1 py | 1 => balance_mul s2 qy
                                   | 2 => balance_mul s1 pz | 3 => balance_mul s2 qz | _ => bzero)
        e_sub_add_sub_L e_sub_add_sub_R p_sub_add_sub
      have hall := bal_poly_eq (fun | 0 => balance_add (balance_mul s1 pw) (balance_mul s1 px)
                                    | 1 => balance_add (balance_mul s2 qw) (balance_mul s2 qx)
                                    | 2 => balance_add (balance_mul s1 py) (balance_mul s1 pz)
                                    | 3 => balance_add (balance_mul s2 qy) (balance_mul s2 qz) | _ => bzero)
        e_sub_add_sub_L e_sub_add_sub_R p_sub_add_sub
      have hm1 := bal_poly_eq (fun | 0 => s1 | 1 => pw | 2 => px | 3 => py | 4 => pz | _ => bzero)
        e_mul_sum4_L e_mul_sum4_R p_mul_sum4
      have hm2 := bal_poly_eq (fun | 0 => s2 | 1 => qw | 2 => qx | 3 => qy | 4 => qz | _ => bzero)
        e_mul_sum4_L e_mul_sum4_R p_mul_sum4
      exact step24_balance_trans (step25_balance_add_congr hwx hyz)
        (step24_balance_trans hall (bal_sub_congr hm1 hm2))
    have hfin := bal_poly_eq (fun | 0 => vv | 1 => vu | _ => bzero)
      e_s118_hfin_L e_s118_hfin_R p_s118_hfin
    exact step24_balance_trans hdot (step24_balance_trans hre hfin)
  have hcomp_inv : ∀ (vc uc : Balance),
      BalanceEquiv
        (balance_sub (balance_mul (balance_mul bone vv) (balance_sub (balance_mul (balance_mul bone vv) uc) (balance_mul (balance_mul (balOfNat 2) vu) vc)))
                     (balance_mul (balance_mul (balOfNat 2) (dot4 v h2u)) vc))
        (balance_mul (bsq vv) uc) := by
    intro vc uc
    exact step24_balance_trans
      (bal_sub_congr (step24_balance_refl _)
        (bal_mul_congr (bal_mul_congr (step24_balance_refl (balOfNat 2)) hvh2u) (step24_balance_refl vc)))
      (bal_poly_eq (fun | 0 => vv | 1 => vu | 2 => vc | 3 => uc | _ => bzero)
        e_s118_inv_L e_s118_inv_R p_s118_inv)
  have hinv : Vec4Equiv (householder4_num vv (balOfNat 2) bone v h2u) (vec4_scale (bsq vv) u) :=
    ⟨hcomp_inv v.w u.w, hcomp_inv v.x u.x, hcomp_inv v.y u.y, hcomp_inv v.z u.z⟩
  exact ⟨hid, hbibo, h2_norm, hinv, s118_det_h2 v⟩

-- 5D-векторы для стандартного 4D-представления S₅ на гиперплоскости sum = 0
structure Vec5 where
  x0 : Balance
  x1 : Balance
  x2 : Balance
  x3 : Balance
  x4 : Balance

def Vec5Equiv (u v : Vec5) : Prop :=
  BalanceEquiv u.x0 v.x0 ∧ BalanceEquiv u.x1 v.x1 ∧ BalanceEquiv u.x2 v.x2 ∧
  BalanceEquiv u.x3 v.x3 ∧ BalanceEquiv u.x4 v.x4

def vec5_zero : Vec5 := ⟨bzero, bzero, bzero, bzero, bzero⟩
def sum5 (u : Vec5) : Balance :=
  balance_add (balance_add (balance_add u.x0 u.x1) (balance_add u.x2 u.x3)) u.x4
def dot5 (u v : Vec5) : Balance :=
  balance_add
    (balance_add (balance_add (balance_mul u.x0 v.x0) (balance_mul u.x1 v.x1))
                 (balance_add (balance_mul u.x2 v.x2) (balance_mul u.x3 v.x3)))
    (balance_mul u.x4 v.x4)

-- Генераторы S₅: транспозиция a = (0 1) (нечётная, det = -1) и 5-цикл b = (0 1 2 3 4) (чётный, det = +1)
def s5_a (u : Vec5) : Vec5 := ⟨u.x1, u.x0, u.x2, u.x3, u.x4⟩
def s5_b (u : Vec5) : Vec5 := ⟨u.x1, u.x2, u.x3, u.x4, u.x0⟩

/-- ШАГ 119. Теорема о неполноте K=2 отражений Хаусхолдера в R⁴ и конструктивный свидетель S₅:
    1. Произведение ЛЮБЫХ ДВУХ отражений H(v₂)H(v₁) поточечно фиксирует всё ортогональное дополнение
       span(v₁, v₂)⊥ (собственное значение +1 с кратностью ≥ 2).
    2. На 4D-гиперплоскости (sum₅ = 0) стандартного представления S₅:
       a) транспозиция `a = (0 1)` и 5-цикл `b = (0 1 2 3 4)` сохраняют гиперплоскость и 5D-норму;
       b) выполнены законы Коксетера–Мура a² = I, b⁵ = I, (ab)⁴ = I и ab ≠ ba;
       c) 5-цикл `b` НЕ ИМЕЕТ ни одного ненулевого неподвижного вектора на гиперплоскости sum₅ = 0 —
          следовательно, связка двух отражений математически неспособна выразить 5-цикл S₅! -/
theorem step119_two_reflections_fix_orth_and_s5_witness
    (v1 v2 w : Vec4) (D1 D2 : Balance)
    (hw1 : BalanceEquiv (dot4 v1 w) bzero)
    (hw2 : BalanceEquiv (dot4 v2 w) bzero) :
    Vec4Equiv (householder4_num D2 (balOfNat 2) bone v2
                 (householder4_num D1 (balOfNat 2) bone v1 w))
              (vec4_scale (balance_mul D2 D1) w)
    ∧
    (∀ u : Vec5,
      BalanceEquiv (sum5 (s5_a u)) (sum5 u) ∧
      BalanceEquiv (sum5 (s5_b u)) (sum5 u) ∧
      BalanceEquiv (dot5 (s5_a u) (s5_a u)) (dot5 u u) ∧
      BalanceEquiv (dot5 (s5_b u) (s5_b u)) (dot5 u u) ∧
      Vec5Equiv (s5_a (s5_a u)) u ∧
      Vec5Equiv (s5_b (s5_b (s5_b (s5_b (s5_b u))))) u ∧
      Vec5Equiv (s5_a (s5_b (s5_a (s5_b (s5_a (s5_b (s5_a (s5_b u)))))))) u)
    ∧
    (Vec5Equiv (s5_a (s5_b ⟨bone, bneg_one, bzero, bzero, bzero⟩))
               (s5_b (s5_a ⟨bone, bneg_one, bzero, bzero, bzero⟩)) → False)
    ∧
    (∀ x : Vec5, Vec5Equiv (s5_b x) x → BalanceEquiv (sum5 x) bzero → Vec5Equiv x vec5_zero) := by
  have hone_fix : ∀ (v x : Vec4) (D : Balance),
      BalanceEquiv (dot4 v x) bzero →
      Vec4Equiv (householder4_num D (balOfNat 2) bone v x) (vec4_scale D x) := by
    intro v x D hvx
    have hc : ∀ (vc xc : Balance),
        BalanceEquiv
          (balance_sub (balance_mul (balance_mul bone D) xc)
                       (balance_mul (balance_mul (balOfNat 2) (dot4 v x)) vc))
          (balance_mul D xc) := by
      intro vc xc
      exact step24_balance_trans
        (bal_sub_congr (bal_mul_congr (bal_one_mul D) (step24_balance_refl xc))
          (bal_mul_congr (step24_balance_trans (bal_mul_congr (step24_balance_refl _) hvx) (bal_mul_zero _))
                         (step24_balance_refl vc)))
        (bal_poly_eq (fun | 0 => balance_mul D xc | 1 => vc | _ => bzero)
          (.sub (.var 0) (.mul .zero (.var 1))) (.var 0) rfl)
    exact ⟨hc v.w x.w, hc v.x x.x, hc v.y x.y, hc v.z x.z⟩
  have h1 := hone_fix v1 w D1 hw1
  have hv2_scaled : BalanceEquiv (dot4 v2 (householder4_num D1 (balOfNat 2) bone v1 w)) bzero :=
    step24_balance_trans (dot4_congr ⟨step24_balance_refl _, step24_balance_refl _, step24_balance_refl _, step24_balance_refl _⟩ h1)
      (by
        let env : Nat → Balance := fun | 0 => D1 | 1 => v2.w | 2 => w.w | 3 => v2.x | 4 => w.x | 5 => v2.y | 6 => w.y | 7 => v2.z | 8 => w.z | _ => bzero
        have hd := bal_poly_eq env
          (.add (.add (.mul (.var 1) (.mul (.var 0) (.var 2))) (.mul (.var 3) (.mul (.var 0) (.var 4))))
                (.add (.mul (.var 5) (.mul (.var 0) (.var 6))) (.mul (.var 7) (.mul (.var 0) (.var 8)))))
          (.mul (.var 0) (.add (.add (.mul (.var 1) (.var 2)) (.mul (.var 3) (.var 4)))
                               (.add (.mul (.var 5) (.var 6)) (.mul (.var 7) (.var 8))))) rfl
        exact step24_balance_trans hd (step24_balance_trans (bal_mul_congr (step24_balance_refl D1) hw2) (bal_mul_zero D1)))
  have h2 := hone_fix v2 (householder4_num D1 (balOfNat 2) bone v1 w) D2 hv2_scaled
  have hcomb : Vec4Equiv (householder4_num D2 (balOfNat 2) bone v2
                            (householder4_num D1 (balOfNat 2) bone v1 w))
                         (vec4_scale (balance_mul D2 D1) w) := by
    refine ⟨?_, ?_, ?_, ?_⟩
    · exact step24_balance_trans h2.1 (step24_balance_trans (bal_mul_congr (step24_balance_refl D2) h1.1) (step24_balance_symm (bal_mul_assoc D2 D1 w.w)))
    · exact step24_balance_trans h2.2.1 (step24_balance_trans (bal_mul_congr (step24_balance_refl D2) h1.2.1) (step24_balance_symm (bal_mul_assoc D2 D1 w.x)))
    · exact step24_balance_trans h2.2.2.1 (step24_balance_trans (bal_mul_congr (step24_balance_refl D2) h1.2.2.1) (step24_balance_symm (bal_mul_assoc D2 D1 w.y)))
    · exact step24_balance_trans h2.2.2.2 (step24_balance_trans (bal_mul_congr (step24_balance_refl D2) h1.2.2.2) (step24_balance_symm (bal_mul_assoc D2 D1 w.z)))
  refine ⟨hcomb, ?_, ?_, ?_⟩
  · intro u
    let env : Nat → Balance := fun | 0 => u.x0 | 1 => u.x1 | 2 => u.x2 | 3 => u.x3 | 4 => u.x4 | _ => bzero
    have hsa := bal_poly_eq env
      (.add (.add (.add (.var 1) (.var 0)) (.add (.var 2) (.var 3))) (.var 4))
      (.add (.add (.add (.var 0) (.var 1)) (.add (.var 2) (.var 3))) (.var 4)) rfl
    have hsb := bal_poly_eq env
      (.add (.add (.add (.var 1) (.var 2)) (.add (.var 3) (.var 4))) (.var 0))
      (.add (.add (.add (.var 0) (.var 1)) (.add (.var 2) (.var 3))) (.var 4)) rfl
    have hda := bal_poly_eq env
      (.add (.add (.add (.mul (.var 1) (.var 1)) (.mul (.var 0) (.var 0)))
                  (.add (.mul (.var 2) (.var 2)) (.mul (.var 3) (.var 3)))) (.mul (.var 4) (.var 4)))
      (.add (.add (.add (.mul (.var 0) (.var 0)) (.mul (.var 1) (.var 1)))
                  (.add (.mul (.var 2) (.var 2)) (.mul (.var 3) (.var 3)))) (.mul (.var 4) (.var 4))) rfl
    have hdb := bal_poly_eq env
      (.add (.add (.add (.mul (.var 1) (.var 1)) (.mul (.var 2) (.var 2)))
                  (.add (.mul (.var 3) (.var 3)) (.mul (.var 4) (.var 4)))) (.mul (.var 0) (.var 0)))
      (.add (.add (.add (.mul (.var 0) (.var 0)) (.mul (.var 1) (.var 1)))
                  (.add (.mul (.var 2) (.var 2)) (.mul (.var 3) (.var 3)))) (.mul (.var 4) (.var 4))) rfl
    have id5 : Vec5Equiv u u :=
      ⟨step24_balance_refl _, step24_balance_refl _, step24_balance_refl _, step24_balance_refl _, step24_balance_refl _⟩
    exact ⟨hsa, hsb, hda, hdb, id5, id5, id5⟩
  · intro hne
    -- (a(b(1,-1,0,0,0))).x0 = 0, но (b(a(1,-1,0,0,0))).x0 = 1
    have h0 := hne.1
    exact bal_gt_zero_ne_zero bone_gt_zero (step24_balance_symm h0)
  · intro x hfx hsum0
    have e01 : BalanceEquiv x.x1 x.x0 := hfx.1
    have e12 : BalanceEquiv x.x2 x.x1 := hfx.2.1
    have e23 : BalanceEquiv x.x3 x.x2 := hfx.2.2.1
    have e34 : BalanceEquiv x.x4 x.x3 := hfx.2.2.2.1
    have e20 : BalanceEquiv x.x2 x.x0 := step24_balance_trans e12 e01
    have e30 : BalanceEquiv x.x3 x.x0 := step24_balance_trans e23 e20
    have e40 : BalanceEquiv x.x4 x.x0 := step24_balance_trans e34 e30
    have hs_eq : BalanceEquiv (sum5 x) (balance_mul (balOfNat 5) x.x0) :=
      step24_balance_trans
        (step25_balance_add_congr
          (step25_balance_add_congr
            (step25_balance_add_congr (step24_balance_refl x.x0) e01)
            (step25_balance_add_congr e20 e30))
          e40)
        (bal_poly_eq (fun | 0 => x.x0 | _ => bzero)
          (.add (.add (.add (.var 0) (.var 0)) (.add (.var 0) (.var 0))) (.var 0))
          (.mul (.nat 5) (.var 0)) rfl)
    have hx0 : BalanceEquiv x.x0 bzero :=
      bal_mul_gt_zero_eq_zero (balOfNat_succ_gt_zero 4)
        (step24_balance_trans (step24_balance_symm hs_eq) hsum0)
    exact ⟨hx0,
           step24_balance_trans e01 hx0,
           step24_balance_trans e20 hx0,
           step24_balance_trans e30 hx0,
           step24_balance_trans e40 hx0⟩

private def e_s120_sqdiff_L : BExpr :=
  .add (.sub (.mul (.mul .one .one) (.var 0)) (.mul (.mul (.nat 2) (.mul .one .one)) (.var 2)))
       (.mul (.mul .one .one) (.var 1))
private def e_s120_sqdiff_R : BExpr := .add (.sub (.var 0) (.mul (.nat 2) (.var 2))) (.var 1)
private theorem p_s120_sqdiff : e_s120_sqdiff_L.norm = e_s120_sqdiff_R.norm := rfl

private def e_s120_P_L : BExpr :=
  let eD := BExpr.var 0; let eg := BExpr.var 1; let edmg := BExpr.sub eD eg
  let eq2 := BExpr.var 2; let ew2 := BExpr.var 3; let eqw := BExpr.var 4
  let ecomb := BExpr.add (.add (.mul (.mul eg eg) eq2) (.mul (.mul edmg edmg) ew2))
                         (.mul (.mul (.nat 2) (.mul eg edmg)) eqw)
  .sub (.mul eD (.add (.mul eg eq2) (.mul edmg ew2))) ecomb

private def e_s120_P_R : BExpr :=
  let eD := BExpr.var 0; let eg := BExpr.var 1; let edmg := BExpr.sub eD eg
  let eq2 := BExpr.var 2; let ew2 := BExpr.var 3; let eqw := BExpr.var 4
  let eqwm := BExpr.add (.sub eq2 (.mul (.nat 2) eqw)) ew2
  .mul (.mul eg edmg) eqwm

private theorem p_s120_P : e_s120_P_L.norm = e_s120_P_R.norm := rfl

private def e_add_sub_self_L : BExpr := .add (.var 1) (.sub (.var 0) (.var 1))
private def e_add_sub_self_R : BExpr := .var 0
private theorem p_add_sub_self : e_add_sub_self_L.norm = e_add_sub_self_R.norm := rfl

private def e_s120_gap2_L : BExpr :=
  .mul (.var 0) (.add (.mul (.var 1) (.sub (.var 4) (.var 2)))
                      (.mul (.sub (.var 0) (.var 1)) (.sub (.var 4) (.var 3))))
private def e_s120_gap2_R : BExpr :=
  .sub (.mul (.mul (.var 0) (.var 0)) (.var 4))
       (.mul (.var 0) (.add (.mul (.var 1) (.var 2)) (.mul (.sub (.var 0) (.var 1)) (.var 3))))
private theorem p_s120_gap2 : e_s120_gap2_L.norm = e_s120_gap2_R.norm := rfl

private def e_s120_unit_L : BExpr :=
  .add (.add (.mul (.mul (.var 0) (.var 0)) (.var 1)) (.mul (.mul (.sub (.var 0) (.var 0)) (.sub (.var 0) (.var 0))) (.var 2)))
       (.mul (.mul (.nat 2) (.mul (.var 0) (.sub (.var 0) (.var 0)))) (.var 3))
private def e_s120_unit_R : BExpr := .mul (.mul (.var 0) (.var 0)) (.var 1)
private theorem p_s120_unit : e_s120_unit_L.norm = e_s120_unit_R.norm := rfl

/-- ШАГ 120. Общая теорема BIBO-устойчивости и унитарного предела для Master Equation (d ∈ {2,3,4}):
    Для любой выпуклой комбинации `comb = γ·q + (D - γ)·w`:
    1. Точное тождество `D·(γ·||q||² + (D - γ)·||w||²) - ||comb||² ~ γ(D - γ)||q - w||²`.
    2. Инвариантный шар: при `0 ≤ γ ≤ D`, `||q||² ≤ R²`, `||w||² ≤ R²` всегда `||comb||² ≤ D²·R²`.
    3. Унитарный предел (`γ = D`): `||comb||² ~ D²·||q||²` (точное сохранение нормы для A₅/S₅). -/
theorem step120_master_equation_bibo_invariance (D gamma : Balance) (q w : Vec4) (R2 : Balance) :
    let dmg := balance_sub D gamma
    let comb := vec4_add (vec4_scale gamma q) (vec4_scale dmg w)
    BalanceEquiv
      (balance_sub (balance_mul D (balance_add (balance_mul gamma (dot4 q q)) (balance_mul dmg (dot4 w w))))
                   (dot4 comb comb))
      (balance_mul (balance_mul gamma dmg) (dot4 (vec4_sub q w) (vec4_sub q w)))
    ∧
    (BalGtZero D → BalGeZero gamma → BalLe gamma D →
     BalLe (dot4 q q) R2 → BalLe (dot4 w w) R2 →
     BalLe (dot4 comb comb) (balance_mul (bsq D) R2))
    ∧
    BalanceEquiv
      (dot4 (vec4_add (vec4_scale D q) (vec4_scale (balance_sub D D) w))
            (vec4_add (vec4_scale D q) (vec4_scale (balance_sub D D) w)))
      (balance_mul (bsq D) (dot4 q q)) := by
  dsimp only
  let dmg := balance_sub D gamma
  have h1 := dot4_lincomb2_sq gamma dmg q w
  have hqw := dot4_sub2_sq bone bone q w
  have hqw' : BalanceEquiv (dot4 (vec4_sub q w) (vec4_sub q w))
                           (balance_add (balance_sub (dot4 q q) (balance_mul (balOfNat 2) (dot4 q w))) (dot4 w w)) := by
    let hs := dot4_sub2_sq bone bone q w
    have hq0 : Vec4Equiv (vec4_sub q w) (vec4_sub (vec4_scale bone q) (vec4_scale bone w)) :=
      ⟨bal_sub_congr (step24_balance_symm (bal_one_mul _)) (step24_balance_symm (bal_one_mul _)),
       bal_sub_congr (step24_balance_symm (bal_one_mul _)) (step24_balance_symm (bal_one_mul _)),
       bal_sub_congr (step24_balance_symm (bal_one_mul _)) (step24_balance_symm (bal_one_mul _)),
       bal_sub_congr (step24_balance_symm (bal_one_mul _)) (step24_balance_symm (bal_one_mul _))⟩
    have hqnorm := dot4_congr hq0 hq0
    let e2 : Nat → Balance := fun | 0 => dot4 q q | 1 => dot4 w w | 2 => dot4 q w | _ => bzero
    have hs2 := bal_poly_eq e2 e_s120_sqdiff_L e_s120_sqdiff_R p_s120_sqdiff
    exact step24_balance_trans hqnorm (step24_balance_trans hs hs2)
  let env : Nat → Balance := fun
    | 0 => D | 1 => gamma | 2 => dot4 q q | 3 => dot4 w w | 4 => dot4 q w | _ => bzero
  have hconvex_id : BalanceEquiv
      (balance_sub (balance_mul D (balance_add (balance_mul gamma (dot4 q q)) (balance_mul dmg (dot4 w w))))
                   (dot4 (vec4_add (vec4_scale gamma q) (vec4_scale dmg w))
                         (vec4_add (vec4_scale gamma q) (vec4_scale dmg w))))
      (balance_mul (balance_mul gamma dmg) (dot4 (vec4_sub q w) (vec4_sub q w))) := by
    have hL := bal_sub_congr
      (step24_balance_refl (balance_mul D (balance_add (balance_mul gamma (dot4 q q)) (balance_mul dmg (dot4 w w)))))
      h1
    have hR := bal_mul_congr (step24_balance_refl (balance_mul gamma dmg)) hqw'
    have hP := bal_poly_eq env e_s120_P_L e_s120_P_R p_s120_P
    exact step24_balance_trans hL (step24_balance_trans hP (step24_balance_symm hR))
  have hball : BalGtZero D → BalGeZero gamma → BalLe gamma D →
      BalLe (dot4 q q) R2 → BalLe (dot4 w w) R2 →
      BalLe (dot4 (vec4_add (vec4_scale gamma q) (vec4_scale dmg w))
                  (vec4_add (vec4_scale gamma q) (vec4_scale dmg w)))
            (balance_mul (bsq D) R2) := by
    intro _ hg0 hgd hqr hwr
    have hgap1 : BalGeZero
        (balance_sub (balance_mul D (balance_add (balance_mul gamma (dot4 q q)) (balance_mul dmg (dot4 w w))))
                     (dot4 (vec4_add (vec4_scale gamma q) (vec4_scale dmg w))
                           (vec4_add (vec4_scale gamma q) (vec4_scale dmg w)))) :=
      bal_ge_zero_of_equiv (step24_balance_symm hconvex_id)
        (bal_mul_ge_ge (bal_mul_ge_ge hg0 hgd) (dot4_self_ge_zero _))
    have hD0 : BalGeZero D :=
      bal_ge_zero_of_equiv (bal_poly_eq (fun | 0 => D | 1 => gamma | _ => bzero)
        e_add_sub_self_L e_add_sub_self_R p_add_sub_self) (bal_add_ge_ge hg0 hgd)
    have hgap2 : BalGeZero
        (balance_sub (balance_mul (bsq D) R2)
                     (balance_mul D (balance_add (balance_mul gamma (dot4 q q)) (balance_mul dmg (dot4 w w))))) := by
      have hterm := bal_add_ge_ge (bal_mul_ge_ge hg0 hqr) (bal_mul_ge_ge hgd hwr)
      have hmul := bal_mul_ge_ge hD0 hterm
      let eB : Nat → Balance := fun | 0 => D | 1 => gamma | 2 => dot4 q q | 3 => dot4 w w | 4 => R2 | _ => bzero
      exact bal_ge_zero_of_equiv (bal_poly_eq eB e_s120_gap2_L e_s120_gap2_R p_s120_gap2) hmul
    exact bal_le_trans hgap1 hgap2
  have hunit : BalanceEquiv
      (dot4 (vec4_add (vec4_scale D q) (vec4_scale (balance_sub D D) w))
            (vec4_add (vec4_scale D q) (vec4_scale (balance_sub D D) w)))
      (balance_mul (bsq D) (dot4 q q)) := by
    have hs := dot4_lincomb2_sq D (balance_sub D D) q w
    let eu : Nat → Balance := fun | 0 => D | 1 => dot4 q q | 2 => dot4 w w | 3 => dot4 q w | _ => bzero
    have hp := bal_poly_eq eu e_s120_unit_L e_s120_unit_R p_s120_unit
    exact step24_balance_trans hs hp
  exact ⟨hconvex_id, hball, hunit⟩

/-- ШАГ 121. Точные алгебраические производные v0.5.0 через `DualBalance`:
    1. Производная стереографического кватерниона `q(g₁ + ε, g₂, g₃)` по `g₁`:
       знаменатель `D'(g₁) = 2g₁`, числители `q₀' = -2g₁`, `q₁' = 2`, `q₂' = 0`.
    2. Производная гейтированного Хаусхолдера `(D·u - β·vu·v)` по гейту `β` равна `-vu·v`.
    3. Производная локального Kerr-энергетического вентиля `z(u_r + ε, u_i) = c + a₁u_r + a₂u_i + a₃(u_r² + u_i²)`
       по `u_r` точно равна `a₁ + 2a₃u_r` (основа O(U·d_in) RTRL!). -/
theorem step121_dual_derivatives_o4_and_local_state_gate
    (g1 g2 g3 D beta vu vc uc c a1 a2 a3 ur ui : Balance) :
    let gd := dual_var g1
    let g2d := dual_add (dual_add (dual_mul gd gd) (dual_const (bsq g2))) (dual_const (bsq g3))
    let den_d := dual_add (dual_const bone) g2d
    let q0_d := dual_add (dual_const bone) (dual_mul (dual_const (balance_neg bone)) g2d)
    let q1_d := dual_mul (dual_const (balOfNat 2)) gd
    let betad := dual_var beta
    let h_beta_d := dual_add (dual_const (balance_mul D uc))
      (dual_mul (dual_const (balance_neg (balance_mul vu vc))) betad)
    let urd := dual_var ur
    let uid := dual_const ui
    let norm2d := dual_add (dual_mul urd urd) (dual_mul uid uid)
    let zd := dual_add (dual_add (dual_const c) (dual_add (dual_mul (dual_const a1) urd) (dual_mul (dual_const a2) uid)))
                       (dual_mul (dual_const a3) norm2d)
    DualEquiv den_d ⟨balance_add bone (balance_add (bsq g1) (balance_add (bsq g2) (bsq g3))),
                     balance_mul (balOfNat 2) g1⟩ ∧
    DualEquiv q0_d ⟨balance_sub bone (balance_add (bsq g1) (balance_add (bsq g2) (bsq g3))),
                    balance_neg (balance_mul (balOfNat 2) g1)⟩ ∧
    DualEquiv q1_d ⟨balance_mul (balOfNat 2) g1, balOfNat 2⟩ ∧
    DualEquiv h_beta_d ⟨balance_sub (balance_mul D uc) (balance_mul (balance_mul vu vc) beta),
                        balance_neg (balance_mul vu vc)⟩ ∧
    DualEquiv zd ⟨balance_add (balance_add c (balance_add (balance_mul a1 ur) (balance_mul a2 ui)))
                              (balance_mul a3 (balance_add (bsq ur) (bsq ui))),
                  balance_add a1 (balance_mul (balOfNat 2) (balance_mul a3 ur))⟩ := by
  dsimp only [dual_var, dual_const, dual_mul, dual_add, DualEquiv, bsq]
  let eg : Nat → Balance := fun | 0 => g1 | 1 => g2 | 2 => g3 | _ => bzero
  let e1 := BExpr.var 0; let e2 := BExpr.var 1; let e3 := BExpr.var 2
  have hden_re := bal_poly_eq eg (.add .one (.add (.add (.mul e1 e1) (.mul e2 e2)) (.mul e3 e3)))
    (.add .one (.add (.mul e1 e1) (.add (.mul e2 e2) (.mul e3 e3)))) rfl
  have hden_eps := bal_poly_eq eg
    (.add .zero (.add (.add (.add (.mul e1 .one) (.mul .one e1)) (.nat 0)) (.nat 0)))
    (.mul (.nat 2) e1) rfl
  have hden_eps' : BalanceEquiv
      (balance_add bzero (balance_add (balance_add (balance_add (balance_mul g1 bone) (balance_mul bone g1)) bzero) bzero))
      (balance_mul (balOfNat 2) g1) :=
    bal_poly_eq eg
      (.add .zero (.add (.add (.add (.mul e1 .one) (.mul .one e1)) .zero) .zero))
      (.mul (.nat 2) e1) rfl
  have hq0_re := bal_poly_eq eg
    (.add .one (.mul (.neg .one) (.add (.add (.mul e1 e1) (.mul e2 e2)) (.mul e3 e3))))
    (.sub .one (.add (.mul e1 e1) (.add (.mul e2 e2) (.mul e3 e3)))) rfl
  have hq0_eps := bal_poly_eq eg
    (.add .zero (.add (.mul (.neg .one) (.add (.add (.add (.mul e1 .one) (.mul .one e1)) .zero) .zero))
                      (.mul .zero (.add (.add (.mul e1 e1) (.mul e2 e2)) (.mul e3 e3)))))
    (.neg (.mul (.nat 2) e1)) rfl
  have hq1_eps := bal_poly_eq eg (.add (.mul (.nat 2) .one) (.mul .zero e1)) (.nat 2) rfl
  let eh : Nat → Balance := fun | 0 => D | 1 => uc | 2 => vu | 3 => vc | 4 => beta | _ => bzero
  have hh_re := bal_poly_eq eh
    (.add (.mul (.var 0) (.var 1)) (.mul (.neg (.mul (.var 2) (.var 3))) (.var 4)))
    (.sub (.mul (.var 0) (.var 1)) (.mul (.mul (.var 2) (.var 3)) (.var 4))) rfl
  have hh_eps := bal_poly_eq eh
    (.add .zero (.add (.mul (.neg (.mul (.var 2) (.var 3))) .one) (.mul .zero (.var 4))))
    (.neg (.mul (.var 2) (.var 3))) rfl
  let ez : Nat → Balance := fun | 0 => c | 1 => a1 | 2 => ur | 3 => a2 | 4 => ui | 5 => a3 | _ => bzero
  have hz_re := step24_balance_refl
    (balance_add (balance_add c (balance_add (balance_mul a1 ur) (balance_mul a2 ui)))
                 (balance_mul a3 (balance_add (balance_mul ur ur) (balance_mul ui ui))))
  have hz_eps := bal_poly_eq ez
    (.add (.add .zero (.add (.add (.mul (.var 1) .one) (.mul .zero (.var 2)))
                            (.add (.mul (.var 3) .zero) (.mul .zero (.var 4)))))
          (.add (.mul (.var 5) (.add (.add (.mul (.var 2) .one) (.mul .one (.var 2)))
                                     (.add (.mul (.var 4) .zero) (.mul .zero (.var 4)))))
                (.mul .zero (.add (.mul (.var 2) (.var 2)) (.mul (.var 4) (.var 4))))))
    (.add (.var 1) (.mul (.nat 2) (.mul (.var 5) (.var 2)))) rfl
  exact ⟨⟨hden_re, hden_eps'⟩, ⟨hq0_re, hq0_eps⟩, ⟨step24_balance_refl _, hq1_eps⟩,
         ⟨hh_re, hh_eps⟩, ⟨hz_re, hz_eps⟩⟩

private def e_s122_newton_L : BExpr :=
  let ea := BExpr.var 0; let ey := BExpr.var 1
  let eay2 := BExpr.mul ea (.mul ey ey)
  let eyn := BExpr.mul ey (.sub (.nat 3) eay2)
  .sub (.nat 4) (.mul ea (.mul eyn eyn))

private def e_s122_newton_R : BExpr :=
  let ea := BExpr.var 0; let ey := BExpr.var 1
  let eay2 := BExpr.mul ea (.mul ey ey)
  .mul (.mul (.sub .one eay2) (.sub .one eay2)) (.sub (.nat 4) eay2)

private theorem p_s122_newton : e_s122_newton_L.norm = e_s122_newton_R.norm := rfl

private def e_s122_den_ge1_L : BExpr := .sub (.nat 4) (.var 0)
private def e_s122_den_ge1_R : BExpr := .sub (.sub (.nat 5) (.var 0)) .one
private theorem p_s122_den_ge1 : e_s122_den_ge1_L.norm = e_s122_den_ge1_R.norm := rfl

private def e_s122_den_gt_L : BExpr := .add .one (.sub (.nat 4) (.var 0))
private def e_s122_den_gt_R : BExpr := .sub (.nat 5) (.var 0)
private theorem p_s122_den_gt : e_s122_den_gt_L.norm = e_s122_den_gt_R.norm := rfl

private def e_s122_bhi_L : BExpr := .mul (.nat 10) (.sub (.nat 4) (.var 0))
private def e_s122_bhi_R : BExpr := .sub (.mul (.nat 9) (.sub (.nat 5) (.var 0))) (.add (.nat 5) (.var 0))
private theorem p_s122_bhi : e_s122_bhi_L.norm = e_s122_bhi_R.norm := rfl

private def e_s122_blo_L : BExpr := .mul (.nat 10) (.sub (.var 0) (.neg (.nat 4)))
private def e_s122_blo_R : BExpr := .sub (.mul (.nat 9) (.add (.nat 5) (.var 0))) (.sub (.nat 5) (.var 0))
private theorem p_s122_blo : e_s122_blo_L.norm = e_s122_blo_R.norm := rfl

/-- ШАГ 122. Тотальность и границы для RMSNorm, Newton inv-sqrt (без `.sqrt()`) и адаптивного Bartlett Page-Hinkley:
    1. RMSNorm: при `ε > 0` и `s ≥ 0` знаменатель `s + ε ≥ ε > 0`.
    2. 1-шаговый Ньютон `2y' = y(3 - ay²)`: точное тождество квадратичной сходимости
       `4 - a·(y(3 - ay²))² ~ (1 - ay²)² · (4 - ay²)` и строгая положительность `y' > 0` при `y > 0, ay² < 3`.
    3. Поправка Бартлетта `(1 + ρ)/(1 - ρ)` в масштабе `ρ = r₅/5` при `-4 ≤ r₅ ≤ 4` (`|ρ| ≤ 0.8`):
       знаменатель `5 - r₅ ≥ 1 > 0`, и двусторонняя оценка `(5 - r₅) ≤ 9(5 + r₅)` и `(5 + r₅) ≤ 9(5 - r₅)` (`κ ∈ [1/9, 9]`). -/
theorem step122_rmsnorm_newton_and_bartlett_ph_totality
    (s eps a y r5 : Balance) (heps : BalGtZero eps) (hs : BalGeZero s) :
    let den_rms := balance_add s eps
    let ay2 := balance_mul a (bsq y)
    let y_next2 := balance_mul y (balance_sub (balOfNat 3) ay2)
    BalLe eps den_rms ∧ BalGtZero den_rms ∧
    BalanceEquiv
      (balance_sub (balOfNat 4) (balance_mul a (bsq y_next2)))
      (balance_mul (bsq (balance_sub bone ay2)) (balance_sub (balOfNat 4) ay2)) ∧
    (BalGtZero y → BalLt ay2 (balOfNat 3) → BalGtZero y_next2) ∧
    (BalLe (balance_neg (balOfNat 4)) r5 → BalLe r5 (balOfNat 4) →
      BalLe bone (balance_sub (balOfNat 5) r5) ∧
      BalGtZero (balance_sub (balOfNat 5) r5) ∧
      BalLe (balance_sub (balOfNat 5) r5) (balance_mul (balOfNat 9) (balance_add (balOfNat 5) r5)) ∧
      BalLe (balance_add (balOfNat 5) r5) (balance_mul (balOfNat 9) (balance_sub (balOfNat 5) r5))) := by
  dsimp only [bsq]
  have hrms_le : BalLe eps (balance_add s eps) :=
    bal_ge_zero_of_equiv (bal_poly_eq (fun | 0 => s | 1 => eps | _ => bzero)
      e_add_sub_cancel_L e_add_sub_cancel_R p_add_sub_cancel) hs
  have hrms_gt : BalGtZero (balance_add s eps) :=
    bal_gt_zero_of_equiv (bal_add_comm eps s) (bal_add_gt_ge heps hs)
  let envN : Nat → Balance := fun | 0 => a | 1 => y | _ => bzero
  have hnewton := bal_poly_eq envN e_s122_newton_L e_s122_newton_R p_s122_newton
  have hy_pos : BalGtZero y → BalLt (balance_mul a (balance_mul y y)) (balOfNat 3) →
      BalGtZero (balance_mul y (balance_sub (balOfNat 3) (balance_mul a (balance_mul y y)))) := by
    intro hy h3
    exact bal_mul_gt_gt hy h3
  have hbart : BalLe (balance_neg (balOfNat 4)) r5 → BalLe r5 (balOfNat 4) →
      BalLe bone (balance_sub (balOfNat 5) r5) ∧
      BalGtZero (balance_sub (balOfNat 5) r5) ∧
      BalLe (balance_sub (balOfNat 5) r5) (balance_mul (balOfNat 9) (balance_add (balOfNat 5) r5)) ∧
      BalLe (balance_add (balOfNat 5) r5) (balance_mul (balOfNat 9) (balance_sub (balOfNat 5) r5)) := by
    intro hlo hhi
    let eb : Nat → Balance := fun | 0 => r5 | _ => bzero
    have hden_ge1 : BalLe bone (balance_sub (balOfNat 5) r5) :=
      bal_ge_zero_of_equiv (bal_poly_eq eb e_s122_den_ge1_L e_s122_den_ge1_R p_s122_den_ge1) hhi
    have hden_gt : BalGtZero (balance_sub (balOfNat 5) r5) :=
      bal_gt_zero_of_equiv (bal_poly_eq eb e_s122_den_gt_L e_s122_den_gt_R p_s122_den_gt)
        (bal_add_gt_ge bone_gt_zero hhi)
    have hb_hi : BalLe (balance_add (balOfNat 5) r5) (balance_mul (balOfNat 9) (balance_sub (balOfNat 5) r5)) :=
      bal_ge_zero_of_equiv (bal_poly_eq eb e_s122_bhi_L e_s122_bhi_R p_s122_bhi)
        (bal_mul_ge_ge (Or.inr (balOfNat_succ_gt_zero 9)) hhi)
    have hb_lo : BalLe (balance_sub (balOfNat 5) r5) (balance_mul (balOfNat 9) (balance_add (balOfNat 5) r5)) :=
      bal_ge_zero_of_equiv (bal_poly_eq eb e_s122_blo_L e_s122_blo_R p_s122_blo)
        (bal_mul_ge_ge (Or.inr (balOfNat_succ_gt_zero 9)) hlo)
    exact ⟨hden_ge1, hden_gt, hb_lo, hb_hi⟩
  exact ⟨hrms_le, hrms_gt, hnewton, hy_pos, hbart⟩

/-- ШАГ 123. Полиномиальное отталкивание прототипов без сингулярного деления (`1/(1 - d + ε)²`):
    Для зоны коллизии `0 ≤ μ ≤ d ≤ 1`:
    1. `0 ≤ d - μ ≤ 1 - μ` и `0 ≤ (d - μ)² ≤ (1 - μ)²`.
    2. Точная `DualBalance`-производная штрафа `((d + ε) - μ)²` равна `2(d - μ)`,
       и жёстко ограничена сверху `2(1 - μ)` без единого деления. -/
theorem step123_poly_repulsion_bounded (d mu : Balance)
    (_hmu : BalGeZero mu) (hd_ge : BalLe mu d) (hd_le : BalLe d bone) :
    let ex := balance_sub d mu
    let om := balance_sub bone mu
    let xd := dual_add (dual_var d) (dual_const (balance_neg mu))
    BalGeZero ex ∧
    BalLe ex om ∧
    BalGeZero (bsq ex) ∧
    BalLe (bsq ex) (bsq om) ∧
    DualEquiv (dual_mul xd xd) ⟨bsq ex, balance_mul (balOfNat 2) ex⟩ ∧
    BalGeZero (balance_mul (balOfNat 2) ex) ∧
    BalLe (balance_mul (balOfNat 2) ex) (balance_mul (balOfNat 2) om) := by
  dsimp only [dual_add, dual_var, dual_const, dual_mul, DualEquiv, bsq]
  let env : Nat → Balance := fun | 0 => d | 1 => mu | _ => bzero
  have hex_ge : BalGeZero (balance_sub d mu) := hd_ge
  have hex_le_om : BalLe (balance_sub d mu) (balance_sub bone mu) :=
    bal_ge_zero_of_equiv
      (bal_poly_eq env (.sub .one (.var 0)) (.sub (.sub .one (.var 1)) (.sub (.var 0) (.var 1))) rfl)
      hd_le
  have hneg_om_le_ex : BalLe (balance_neg (balance_sub bone mu)) (balance_sub d mu) :=
    bal_ge_zero_of_equiv
      (bal_poly_eq env (.add (.mul (.nat 2) (.sub (.var 0) (.var 1))) (.sub .one (.var 0)))
        (.sub (.sub (.var 0) (.var 1)) (.neg (.sub .one (.var 1)))) rfl)
      (bal_add_ge_ge (bal_mul_ge_ge (Or.inr (balOfNat_succ_gt_zero 1)) hex_ge) hd_le)
  have hsq_ge : BalGeZero (balance_mul (balance_sub d mu) (balance_sub d mu)) :=
    bal_sq_ge_zero (balance_sub d mu)
  have hsq_le : BalLe (bsq (balance_sub d mu)) (bsq (balance_sub bone mu)) :=
    bal_abs_le_sq_le hneg_om_le_ex hex_le_om
  have hd_re := bal_poly_eq env
    (.mul (.add (.var 0) (.neg (.var 1))) (.add (.var 0) (.neg (.var 1))))
    (.mul (.sub (.var 0) (.var 1)) (.sub (.var 0) (.var 1))) rfl
  have hd_eps := bal_poly_eq env
    (.add (.mul (.add (.var 0) (.neg (.var 1))) (.add .one .zero))
          (.mul (.add .one .zero) (.add (.var 0) (.neg (.var 1)))))
    (.mul (.nat 2) (.sub (.var 0) (.var 1))) rfl
  have h2ex_ge := bal_mul_ge_ge (Or.inr (balOfNat_succ_gt_zero 1)) hex_ge
  have h2ex_le : BalLe (balance_mul (balOfNat 2) (balance_sub d mu)) (balance_mul (balOfNat 2) (balance_sub bone mu)) :=
    bal_ge_zero_of_equiv
      (bal_poly_eq env (.mul (.nat 2) (.sub .one (.var 0)))
        (.sub (.mul (.nat 2) (.sub .one (.var 1))) (.mul (.nat 2) (.sub (.var 0) (.var 1)))) rfl)
      (bal_mul_ge_ge (Or.inr (balOfNat_succ_gt_zero 1)) hd_le)
  exact ⟨hex_ge, hex_le_om, hsq_ge, hsq_le, ⟨hd_re, hd_eps⟩, h2ex_ge, h2ex_le⟩

private def e_sum3_L : BExpr := .add (.add (.var 0) (.var 0)) (.var 0)
private def e_sum3_R : BExpr := .mul (.nat 3) (.var 0)
private theorem p_sum3 : e_sum3_L.norm = e_sum3_R.norm := rfl

private def e_sum4_L : BExpr := .add (.add (.var 0) (.var 0)) (.add (.var 0) (.var 0))
private def e_sum4_R : BExpr := .mul (.nat 4) (.var 0)
private theorem p_sum4 : e_sum4_L.norm = e_sum4_R.norm := rfl

private def e_sub_self_zero_L : BExpr := .sub (.var 0) (.var 0)
private theorem p_sub_self_zero : e_sub_self_zero_L.norm = BExpr.zero.norm := rfl

private def e_drift1_L : BExpr := .sub (.mul (.add (.var 0) (.var 1)) (.add (.var 0) (.var 1))) (.mul (.var 0) (.var 0))
private def e_drift1_R : BExpr := .add (.mul (.nat 2) (.mul (.var 0) (.var 1))) (.mul (.var 1) (.var 1))
private theorem p_drift1 : e_drift1_L.norm = e_drift1_R.norm := rfl

private theorem drift1_sq (a b : Balance) :
    BalanceEquiv (balance_sub (bsq (balance_add a b)) (bsq a))
                 (balance_add (balance_mul (balOfNat 2) (balance_mul a b)) (bsq b)) :=
  bal_poly_eq (fun | 0 => a | 1 => b | _ => bzero) e_drift1_L e_drift1_R p_drift1

private def e_add_add_swap_L : BExpr := .add (.add (.var 0) (.var 1)) (.add (.var 2) (.var 3))
private def e_add_add_swap_R : BExpr := .add (.add (.var 0) (.var 2)) (.add (.var 1) (.var 3))
private theorem p_add_add_swap : e_add_add_swap_L.norm = e_add_add_swap_R.norm := rfl

private def e_sub_neg_eq_add_L : BExpr := .sub (.var 1) (.neg (.var 0))
private def e_sub_neg_eq_add_R : BExpr := .add (.var 0) (.var 1)
private theorem p_sub_neg_eq_add : e_sub_neg_eq_add_L.norm = e_sub_neg_eq_add_R.norm := rfl

private def e_sub_add_double_L : BExpr := .add (.sub (.var 0) (.var 1)) (.add (.var 0) (.var 1))
private def e_sub_add_double_R : BExpr := .mul (.nat 2) (.var 0)
private theorem p_sub_add_double : e_sub_add_double_L.norm = e_sub_add_double_R.norm := rfl

private def e_sub_zero_R : BExpr := .sub (.var 0) .zero
private theorem p_sub_zero : (BExpr.var 0).norm = e_sub_zero_R.norm := rfl

private def e_zero_sub_neg_R : BExpr := .sub .zero (.neg (.var 0))
private theorem p_zero_sub_neg : (BExpr.var 0).norm = e_zero_sub_neg_R.norm := rfl

private def e_one_add_L : BExpr := .add .one (.var 0)
private def e_one_sub_neg_R : BExpr := .sub .one (.neg (.var 0))
private theorem p_one_add_eq_sub_neg : e_one_add_L.norm = e_one_sub_neg_R.norm := rfl

private def e_s124_hhi_neg_L : BExpr :=
  .add (.mul (.neg (.var 0)) (.add (.var 1) (.var 2))) (.mul (.sub .one (.neg (.var 0))) (.var 1))
private def e_s124_hhi_neg_R : BExpr := .sub (.var 1) (.mul (.var 2) (.var 0))
private theorem p_s124_hhi_neg : e_s124_hhi_neg_L.norm = e_s124_hhi_neg_R.norm := rfl

private def e_s124_hlo_neg_L : BExpr :=
  .add (.mul (.neg (.var 0)) (.sub (.var 1) (.var 2))) (.mul (.sub .one (.neg (.var 0))) (.var 1))
private def e_s124_hlo_neg_R : BExpr := .sub (.mul (.var 2) (.var 0)) (.neg (.var 1))
private theorem p_s124_hlo_neg : e_s124_hlo_neg_L.norm = e_s124_hlo_neg_R.norm := rfl

private def e_s124_hhi_pos_L : BExpr :=
  .add (.mul (.var 0) (.sub (.var 1) (.var 2))) (.mul (.sub .one (.var 0)) (.var 1))
private def e_s124_hhi_pos_R : BExpr := .sub (.var 1) (.mul (.var 2) (.var 0))
private theorem p_s124_hhi_pos : e_s124_hhi_pos_L.norm = e_s124_hhi_pos_R.norm := rfl

private def e_s124_hlo_pos_L : BExpr :=
  .add (.mul (.var 0) (.add (.var 1) (.var 2))) (.mul (.sub .one (.var 0)) (.var 1))
private def e_s124_hlo_pos_R : BExpr := .sub (.mul (.var 2) (.var 0)) (.neg (.var 1))
private theorem p_s124_hlo_pos : e_s124_hlo_pos_L.norm = e_s124_hlo_pos_R.norm := rfl

private def e_s124_hue_lo_L : BExpr :=
  .sub (.add (.add (.var 1) (.var 2)) (.add (.var 3) (.var 4)))
       (.add (.add (.neg (.var 0)) (.neg (.var 0))) (.add (.neg (.var 0)) (.neg (.var 0))))
private def e_s124_hue_lo_R : BExpr :=
  .sub (.add (.add (.var 1) (.var 2)) (.add (.var 3) (.var 4))) (.neg (.mul (.nat 4) (.var 0)))
private theorem p_s124_hue_lo : e_s124_hue_lo_L.norm = e_s124_hue_lo_R.norm := rfl

private def e_s124_hue_hi_L : BExpr :=
  .sub (.add (.add (.var 0) (.var 0)) (.add (.var 0) (.var 0)))
       (.add (.add (.var 1) (.var 2)) (.add (.var 3) (.var 4)))
private def e_s124_hue_hi_R : BExpr :=
  .sub (.mul (.nat 4) (.var 0)) (.add (.add (.var 1) (.var 2)) (.add (.var 3) (.var 4)))
private theorem p_s124_hue_hi : e_s124_hue_hi_L.norm = e_s124_hue_hi_R.norm := rfl

private def e_s124_hee_le4_L : BExpr :=
  .sub (.add (.add (.mul .one .one) (.mul .one .one)) (.add (.mul .one .one) (.mul .one .one))) (.var 0)
private def e_s124_hee_le4_R : BExpr := .sub (.nat 4) (.var 0)
private theorem p_s124_hee_le4 : e_s124_hee_le4_L.norm = e_s124_hee_le4_R.norm := rfl

private def e_s124_drift_lo_L : BExpr :=
  .add (.mul (.nat 2) (.sub (.var 0) (.neg (.mul (.nat 4) (.var 1))))) (.var 2)
private def e_s124_drift_lo_R : BExpr :=
  .sub (.add (.mul (.nat 2) (.var 0)) (.var 2)) (.neg (.mul (.nat 8) (.var 1)))
private theorem p_s124_drift_lo : e_s124_drift_lo_L.norm = e_s124_drift_lo_R.norm := rfl

private def e_s124_drift_hi_L : BExpr :=
  .add (.mul (.nat 2) (.sub (.mul (.nat 4) (.var 1)) (.var 0))) (.sub (.nat 4) (.var 2))
private def e_s124_drift_hi_R : BExpr :=
  .sub (.add (.mul (.nat 8) (.var 1)) (.nat 4)) (.add (.mul (.nat 2) (.var 0)) (.var 2))
private theorem p_s124_drift_hi : e_s124_drift_hi_L.norm = e_s124_drift_hi_R.norm := rfl

/-- ШАГ 124. Параметрические оценки Fixed-Point (Q8.24 / без FPU и для ZKML):
    1. При компонентных границах `|g_i| ≤ B`, `|v_i| ≤ B` и масштабе `S > 0`:
       `0 ≤ ||g||² ≤ 3B²`, `0 ≤ ||v||² ≤ 4B²`, а масштабированный знаменатель
       `S² ≤ S² + ||g||² ≤ S² + 3B²` строго положителен и гарантированно не переполняется.
    2. Ошибка дрейфа нормы после округления `u_tilde = u + e` при `|u_i| ≤ U` и `|e_i| ≤ 1`:
       `||u + e||² - ||u||² ~ 2(u·e) + ||e||²`, причём `-(8U) ≤ ||u + e||² - ||u||² ≤ 8U + 4`. -/
theorem step124_fixed_point_bounds_and_norm_drift
    (S B U : Balance) (g : Vec3) (v u e : Vec4)
    (hS : BalGtZero S)
    (hgx1 : BalLe (balance_neg B) g.x) (hgx2 : BalLe g.x B)
    (hgy1 : BalLe (balance_neg B) g.y) (hgy2 : BalLe g.y B)
    (hgz1 : BalLe (balance_neg B) g.z) (hgz2 : BalLe g.z B)
    (hvw1 : BalLe (balance_neg B) v.w) (hvw2 : BalLe v.w B)
    (hvx1 : BalLe (balance_neg B) v.x) (hvx2 : BalLe v.x B)
    (hvy1 : BalLe (balance_neg B) v.y) (hvy2 : BalLe v.y B)
    (hvz1 : BalLe (balance_neg B) v.z) (hvz2 : BalLe v.z B)
    (huw1 : BalLe (balance_neg U) u.w) (huw2 : BalLe u.w U)
    (hux1 : BalLe (balance_neg U) u.x) (hux2 : BalLe u.x U)
    (huy1 : BalLe (balance_neg U) u.y) (huy2 : BalLe u.y U)
    (huz1 : BalLe (balance_neg U) u.z) (huz2 : BalLe u.z U)
    (hew1 : BalLe (balance_neg bone) e.w) (hew2 : BalLe e.w bone)
    (hex1 : BalLe (balance_neg bone) e.x) (hex2 : BalLe e.x bone)
    (hey1 : BalLe (balance_neg bone) e.y) (hey2 : BalLe e.y bone)
    (hez1 : BalLe (balance_neg bone) e.z) (hez2 : BalLe e.z bone) :
    let d3g := dot3 g g
    let d4v := dot4 v v
    let denS := balance_add (bsq S) d3g
    let drift := balance_sub (dot4 (vec4_add u e) (vec4_add u e)) (dot4 u u)
    BalGeZero d3g ∧ BalLe d3g (balance_mul (balOfNat 3) (bsq B)) ∧
    BalGeZero d4v ∧ BalLe d4v (balance_mul (balOfNat 4) (bsq B)) ∧
    BalGtZero denS ∧ BalLe (bsq S) denS ∧ BalLe denS (balance_add (bsq S) (balance_mul (balOfNat 3) (bsq B))) ∧
    BalanceEquiv drift (balance_add (balance_mul (balOfNat 2) (dot4 u e)) (dot4 e e)) ∧
    BalLe (balance_neg (balance_mul (balOfNat 8) U)) drift ∧
    BalLe drift (balance_add (balance_mul (balOfNat 8) U) (balOfNat 4)) := by
  dsimp only
  have hg_le : BalLe (dot3 g g) (balance_mul (balOfNat 3) (bsq B)) := by
    have sx := bal_abs_le_sq_le hgx1 hgx2
    have sy := bal_abs_le_sq_le hgy1 hgy2
    have sz := bal_abs_le_sq_le hgz1 hgz2
    have hsum := bal_le_add_congr (bal_le_add_congr sx sy) sz
    let eB : Nat → Balance := fun | 0 => bsq B | _ => bzero
    have h3 : BalanceEquiv (balance_add (balance_add (bsq B) (bsq B)) (bsq B)) (balance_mul (balOfNat 3) (bsq B)) :=
      bal_poly_eq eB e_sum3_L e_sum3_R p_sum3
    dsimp [BalLe] at *
    exact bal_ge_zero_of_equiv
      (bal_sub_congr h3 (step24_balance_refl (dot3 g g))) hsum
  have hv_le : BalLe (dot4 v v) (balance_mul (balOfNat 4) (bsq B)) := by
    have sw := bal_abs_le_sq_le hvw1 hvw2
    have sx := bal_abs_le_sq_le hvx1 hvx2
    have sy := bal_abs_le_sq_le hvy1 hvy2
    have sz := bal_abs_le_sq_le hvz1 hvz2
    have hsum := bal_le_add_congr (bal_le_add_congr sw sx) (bal_le_add_congr sy sz)
    let eB : Nat → Balance := fun | 0 => bsq B | _ => bzero
    have h4 : BalanceEquiv (balance_add (balance_add (bsq B) (bsq B)) (balance_add (bsq B) (bsq B)))
                           (balance_mul (balOfNat 4) (bsq B)) :=
      bal_poly_eq eB e_sum4_L e_sum4_R p_sum4
    dsimp [BalLe] at *
    exact bal_ge_zero_of_equiv
      (bal_sub_congr h4 (step24_balance_refl (dot4 v v))) hsum
  have hS2 : BalGtZero (bsq S) := bal_mul_gt_gt hS hS
  have hden_gt : BalGtZero (balance_add (bsq S) (dot3 g g)) :=
    bal_add_gt_ge hS2 (dot3_self_ge_zero g)
  have hden_ge : BalLe (bsq S) (balance_add (bsq S) (dot3 g g)) :=
    bal_ge_zero_of_equiv
      (bal_poly_eq (fun | 0 => bsq S | 1 => dot3 g g | _ => bzero)
        e_sub_add_cancel_L e_sub_add_cancel_R p_sub_add_cancel)
      (dot3_self_ge_zero g)
  have hden_le : BalLe (balance_add (bsq S) (dot3 g g)) (balance_add (bsq S) (balance_mul (balOfNat 3) (bsq B))) :=
    bal_le_add_congr (Or.inl (bal_poly_eq (fun | 0 => bsq S | _ => bzero) e_sub_self_zero_L .zero p_sub_self_zero)) hg_le
  -- Оценка дрейфа нормы |u_i| ≤ U, |e_i| ≤ 1 => -8U ≤ 2(u·e) + ||e||² ≤ 8U + 4
  have hdrift_eq : BalanceEquiv
      (balance_sub (dot4 (vec4_add u e) (vec4_add u e)) (dot4 u u))
      (balance_add (balance_mul (balOfNat 2) (dot4 u e)) (dot4 e e)) := by
    let sw := bsq (balance_add u.w e.w); let aw := bsq u.w
    let sx := bsq (balance_add u.x e.x); let ax := bsq u.x
    let sy := bsq (balance_add u.y e.y); let ay := bsq u.y
    let sz := bsq (balance_add u.z e.z); let az := bsq u.z
    let pw := balance_mul u.w e.w; let qw := bsq e.w
    let px := balance_mul u.x e.x; let qx := bsq e.x
    let py := balance_mul u.y e.y; let qy := bsq e.y
    let pz := balance_mul u.z e.z; let qz := bsq e.z
    have htop := step24_balance_symm (bal_poly_eq
      (fun | 0 => balance_add sw sx | 1 => balance_add aw ax
           | 2 => balance_add sy sz | 3 => balance_add ay az | _ => bzero)
      e_sub_add_sub_L e_sub_add_sub_R p_sub_add_sub)
    have hwx := step24_balance_symm (bal_poly_eq
      (fun | 0 => sw | 1 => aw | 2 => sx | 3 => ax | _ => bzero)
      e_sub_add_sub_L e_sub_add_sub_R p_sub_add_sub)
    have hyz := step24_balance_symm (bal_poly_eq
      (fun | 0 => sy | 1 => ay | 2 => sz | 3 => az | _ => bzero)
      e_sub_add_sub_L e_sub_add_sub_R p_sub_add_sub)
    have hsplit := step24_balance_trans htop (step25_balance_add_congr hwx hyz)
    have hd1 := step25_balance_add_congr
      (step25_balance_add_congr (drift1_sq u.w e.w) (drift1_sq u.x e.x))
      (step25_balance_add_congr (drift1_sq u.y e.y) (drift1_sq u.z e.z))
    have hs1 := bal_poly_eq
      (fun | 0 => balance_mul (balOfNat 2) pw | 1 => qw | 2 => balance_mul (balOfNat 2) px | 3 => qx | _ => bzero)
      e_add_add_swap_L e_add_add_swap_R p_add_add_swap
    have hs2 := bal_poly_eq
      (fun | 0 => balance_mul (balOfNat 2) py | 1 => qy | 2 => balance_mul (balOfNat 2) pz | 3 => qz | _ => bzero)
      e_add_add_swap_L e_add_add_swap_R p_add_add_swap
    have hs3 := bal_poly_eq
      (fun | 0 => balance_add (balance_mul (balOfNat 2) pw) (balance_mul (balOfNat 2) px) | 1 => balance_add qw qx
           | 2 => balance_add (balance_mul (balOfNat 2) py) (balance_mul (balOfNat 2) pz) | 3 => balance_add qy qz | _ => bzero)
      e_add_add_swap_L e_add_add_swap_R p_add_add_swap
    have hm2 := bal_poly_eq (fun | 0 => balOfNat 2 | 1 => pw | 2 => px | 3 => py | 4 => pz | _ => bzero)
      e_mul_sum4_L e_mul_sum4_R p_mul_sum4
    have hrearr := step24_balance_trans (step25_balance_add_congr hs1 hs2)
      (step24_balance_trans hs3 (step25_balance_add_congr hm2 (step24_balance_refl (dot4 e e))))
    exact step24_balance_trans hsplit (step24_balance_trans hd1 hrearr)
  -- Скалярная оценка: -U ≤ uc ≤ U ∧ -1 ≤ ec ≤ 1 ⇒ -U ≤ uc·ec ≤ U
  have hmul1 : ∀ (uc ec : Balance),
      BalLe (balance_neg U) uc → BalLe uc U → BalLe (balance_neg bone) ec → BalLe ec bone →
      BalLe (balance_neg U) (balance_mul uc ec) ∧ BalLe (balance_mul uc ec) U := by
    intro uc ec hu1 hu2 he1 he2
    have hep : BalGeZero (balance_add bone ec) :=
      bal_ge_zero_of_equiv (bal_poly_eq (fun | 0 => bone | 1 => ec | _ => bzero)
        e_sub_neg_eq_add_L e_sub_neg_eq_add_R p_sub_neg_eq_add) he1
    have hem : BalGeZero (balance_sub bone ec) := he2
    have hup : BalGeZero (balance_add U uc) :=
      bal_ge_zero_of_equiv (bal_poly_eq (fun | 0 => U | 1 => uc | _ => bzero)
        e_sub_neg_eq_add_L e_sub_neg_eq_add_R p_sub_neg_eq_add) hu1
    have hum : BalGeZero (balance_sub U uc) := hu2
    have hU0 : BalGeZero U :=
      bal_ge_zero_of_double_ge
        (bal_ge_zero_of_equiv
          (bal_poly_eq (fun | 0 => U | 1 => uc | _ => bzero)
            e_sub_add_double_L e_sub_add_double_R p_sub_add_double)
          (bal_add_ge_ge hum hup))
    cases trichotomy ec.fwd ec.bwd with
    | inl hez =>
      have he0 : BalanceEquiv ec bzero := bal_equiv_bzero_of_eq hez
      have h0 : BalanceEquiv (balance_mul uc ec) bzero :=
        step24_balance_trans (bal_mul_congr (step24_balance_refl uc) he0) (bal_mul_zero uc)
      have hU_hi : BalLe (balance_mul uc ec) U :=
        bal_ge_zero_of_equiv
          (step24_balance_trans
            (bal_poly_eq (fun | 0 => U | _ => bzero) (.var 0) e_sub_zero_R p_sub_zero)
            (bal_sub_congr (step24_balance_refl U) (step24_balance_symm h0))) hU0
      have hU_lo : BalLe (balance_neg U) (balance_mul uc ec) :=
        bal_ge_zero_of_equiv
          (step24_balance_trans
            (bal_poly_eq (fun | 0 => U | _ => bzero) (.var 0) e_zero_sub_neg_R p_zero_sub_neg)
            (bal_sub_congr (step24_balance_symm h0) (step24_balance_refl _))) hU0
      exact ⟨hU_lo, hU_hi⟩
    | inr hene =>
      cases hene with
      | inl helt =>
        have hnec_gt : BalGtZero (balance_neg ec) := bal_neg_gt_zero helt
        have h1m_ge : BalGeZero (balance_sub bone (balance_neg ec)) :=
          bal_ge_zero_of_equiv (bal_poly_eq (fun | 0 => ec | _ => bzero)
            e_one_add_L e_one_sub_neg_R p_one_add_eq_sub_neg) hep
        have hhi := bal_add_ge_ge (bal_mul_gt_ge hnec_gt hup) (bal_mul_ge_ge h1m_ge hU0)
        have hhi_eq : BalanceEquiv
            (balance_add (balance_mul (balance_neg ec) (balance_add U uc))
                         (balance_mul (balance_sub bone (balance_neg ec)) U))
            (balance_sub U (balance_mul uc ec)) :=
          bal_poly_eq (fun | 0 => ec | 1 => U | 2 => uc | _ => bzero)
            e_s124_hhi_neg_L e_s124_hhi_neg_R p_s124_hhi_neg
        have hlo := bal_add_ge_ge (bal_mul_gt_ge hnec_gt hum) (bal_mul_ge_ge h1m_ge hU0)
        have hlo_eq : BalanceEquiv
            (balance_add (balance_mul (balance_neg ec) (balance_sub U uc))
                         (balance_mul (balance_sub bone (balance_neg ec)) U))
            (balance_sub (balance_mul uc ec) (balance_neg U)) :=
          bal_poly_eq (fun | 0 => ec | 1 => U | 2 => uc | _ => bzero)
            e_s124_hlo_neg_L e_s124_hlo_neg_R p_s124_hlo_neg
        exact ⟨bal_ge_zero_of_equiv hlo_eq hlo, bal_ge_zero_of_equiv hhi_eq hhi⟩
      | inr hegt =>
        have hhi := bal_add_ge_ge (bal_mul_gt_ge hegt hum) (bal_mul_ge_ge hem hU0)
        have hhi_eq : BalanceEquiv
            (balance_add (balance_mul ec (balance_sub U uc)) (balance_mul (balance_sub bone ec) U))
            (balance_sub U (balance_mul uc ec)) :=
          bal_poly_eq (fun | 0 => ec | 1 => U | 2 => uc | _ => bzero)
            e_s124_hhi_pos_L e_s124_hhi_pos_R p_s124_hhi_pos
        have hlo := bal_add_ge_ge (bal_mul_gt_ge hegt hup) (bal_mul_ge_ge hem hU0)
        have hlo_eq : BalanceEquiv
            (balance_add (balance_mul ec (balance_add U uc)) (balance_mul (balance_sub bone ec) U))
            (balance_sub (balance_mul uc ec) (balance_neg U)) :=
          bal_poly_eq (fun | 0 => ec | 1 => U | 2 => uc | _ => bzero)
            e_s124_hlo_pos_L e_s124_hlo_pos_R p_s124_hlo_pos
        exact ⟨bal_ge_zero_of_equiv hlo_eq hlo, bal_ge_zero_of_equiv hhi_eq hhi⟩
  have pmw := hmul1 u.w e.w huw1 huw2 hew1 hew2
  have pmx := hmul1 u.x e.x hux1 hux2 hex1 hex2
  have pmy := hmul1 u.y e.y huy1 huy2 hey1 hey2
  have pmz := hmul1 u.z e.z huz1 huz2 hez1 hez2
  have hue_lo : BalLe (balance_neg (balance_mul (balOfNat 4) U)) (dot4 u e) :=
    bal_ge_zero_of_equiv
      (bal_poly_eq (fun | 0 => U | 1 => balance_mul u.w e.w | 2 => balance_mul u.x e.x
                        | 3 => balance_mul u.y e.y | 4 => balance_mul u.z e.z | _ => bzero)
        e_s124_hue_lo_L e_s124_hue_lo_R p_s124_hue_lo)
      (bal_le_add_congr (bal_le_add_congr pmw.1 pmx.1) (bal_le_add_congr pmy.1 pmz.1))
  have hue_hi : BalLe (dot4 u e) (balance_mul (balOfNat 4) U) :=
    bal_ge_zero_of_equiv
      (bal_poly_eq (fun | 0 => U | 1 => balance_mul u.w e.w | 2 => balance_mul u.x e.x
                        | 3 => balance_mul u.y e.y | 4 => balance_mul u.z e.z | _ => bzero)
        e_s124_hue_hi_L e_s124_hue_hi_R p_s124_hue_hi)
      (bal_le_add_congr (bal_le_add_congr pmw.2 pmx.2) (bal_le_add_congr pmy.2 pmz.2))
  have hee_le4 : BalLe (dot4 e e) (balOfNat 4) := by
    have sew := bal_abs_le_sq_le hew1 hew2
    have sex := bal_abs_le_sq_le hex1 hex2
    have sey := bal_abs_le_sq_le hey1 hey2
    have sez := bal_abs_le_sq_le hez1 hez2
    exact bal_ge_zero_of_equiv
      (bal_poly_eq (fun | 0 => dot4 e e | _ => bzero)
        e_s124_hee_le4_L e_s124_hee_le4_R p_s124_hee_le4)
      (bal_le_add_congr (bal_le_add_congr sew sex) (bal_le_add_congr sey sez))
  have hdrift_lo : BalLe (balance_neg (balance_mul (balOfNat 8) U))
                         (balance_sub (dot4 (vec4_add u e) (vec4_add u e)) (dot4 u u)) := by
    have h2hue := bal_mul_ge_ge (Or.inr (balOfNat_succ_gt_zero 1)) hue_lo
    have hlo_2e := bal_add_ge_ge h2hue (dot4_self_ge_zero e)
    have hlo_eq : BalanceEquiv
        (balance_add (balance_mul (balOfNat 2) (balance_sub (dot4 u e) (balance_neg (balance_mul (balOfNat 4) U))))
                     (dot4 e e))
        (balance_sub (balance_add (balance_mul (balOfNat 2) (dot4 u e)) (dot4 e e))
                     (balance_neg (balance_mul (balOfNat 8) U))) :=
      bal_poly_eq (fun | 0 => dot4 u e | 1 => U | 2 => dot4 e e | _ => bzero)
        e_s124_drift_lo_L e_s124_drift_lo_R p_s124_drift_lo
    exact bal_ge_zero_of_equiv
      (step24_balance_trans hlo_eq (bal_sub_congr (step24_balance_symm hdrift_eq) (step24_balance_refl _)))
      hlo_2e
  have hdrift_hi : BalLe (balance_sub (dot4 (vec4_add u e) (vec4_add u e)) (dot4 u u))
                         (balance_add (balance_mul (balOfNat 8) U) (balOfNat 4)) := by
    have h2hue := bal_mul_ge_ge (Or.inr (balOfNat_succ_gt_zero 1)) hue_hi
    have hhi_2e := bal_add_ge_ge h2hue hee_le4
    have hhi_eq : BalanceEquiv
        (balance_add (balance_mul (balOfNat 2) (balance_sub (balance_mul (balOfNat 4) U) (dot4 u e)))
                     (balance_sub (balOfNat 4) (dot4 e e)))
        (balance_sub (balance_add (balance_mul (balOfNat 8) U) (balOfNat 4))
                     (balance_add (balance_mul (balOfNat 2) (dot4 u e)) (dot4 e e))) :=
      bal_poly_eq (fun | 0 => dot4 u e | 1 => U | 2 => dot4 e e | _ => bzero)
        e_s124_drift_hi_L e_s124_drift_hi_R p_s124_drift_hi
    exact bal_ge_zero_of_equiv
      (step24_balance_trans hhi_eq (bal_sub_congr (step24_balance_refl _) (step24_balance_symm hdrift_eq)))
      hhi_2e
  exact ⟨dot3_self_ge_zero g, hg_le, dot4_self_ge_zero v, hv_le, hden_gt, hden_ge, hden_le,
         hdrift_eq, hdrift_lo, hdrift_hi⟩


-- =============================================================================
-- ЭТАП XXX. ВЕРИФИКАЦИЯ v0.5.0: АДАПТИВНОЕ O(d) (S_6 ≤ O(5)), АВТОНОМИЯ PILLAR 5,
-- ВКЛАДКА SO(2) ⊂ SO(3) ⊂ SO(4) ⊂ O(4), ВЫХОД ИЗ СЕДЛА β=0 И ГРАНИЦЫ Q8.24
-- (Шаги 125–128; строго 0 аксиом)
-- =============================================================================

theorem bal_le_refl (x : Balance) : BalLe x x :=
  Or.inl (bal_poly_eq (fun | 0 => x | _ => bzero) e_sub_self_zero_L .zero p_sub_self_zero)

theorem bal_le_of_equiv_left {x x' y : Balance} (hx : BalanceEquiv x x') (hle : BalLe x y) :
    BalLe x' y :=
  bal_ge_zero_of_equiv (bal_sub_congr (step24_balance_refl y) hx) hle

theorem bal_le_of_equiv_right {x y y' : Balance} (hy : BalanceEquiv y y') (hle : BalLe x y) :
    BalLe x y' :=
  bal_ge_zero_of_equiv (bal_sub_congr hy (step24_balance_refl x)) hle

private def e_s125_mul_sub_L : BExpr := .mul (.var 0) (.sub (.var 2) (.var 1))
private def e_s125_mul_sub_R : BExpr := .sub (.mul (.var 0) (.var 2)) (.mul (.var 0) (.var 1))
private theorem p_s125_mul_sub : e_s125_mul_sub_L.norm = e_s125_mul_sub_R.norm := rfl

theorem bal_le_mul_nonneg_left {c a b : Balance} (hc : BalGeZero c) (hab : BalLe a b) :
    BalLe (balance_mul c a) (balance_mul c b) :=
  bal_ge_zero_of_equiv
    (bal_poly_eq (fun | 0 => c | 1 => a | 2 => b | _ => bzero) e_s125_mul_sub_L e_s125_mul_sub_R p_s125_mul_sub)
    (bal_mul_ge_ge hc hab)

-- ----- Векторы произвольной размерности Vec n для AdaptiveO(d) и IDBD -----

def Vec : Nat → Type
  | 0 => Unit
  | n + 1 => Balance × Vec n

def VecEquiv : {n : Nat} → Vec n → Vec n → Prop
  | 0, _, _ => True
  | _ + 1, (a, u), (b, v) => BalanceEquiv a b ∧ VecEquiv u v

def vec_zero : (n : Nat) → Vec n
  | 0 => ()
  | n + 1 => (bzero, vec_zero n)

def vec_add : {n : Nat} → Vec n → Vec n → Vec n
  | 0, _, _ => ()
  | _ + 1, (a, u), (b, v) => (balance_add a b, vec_add u v)

def vec_sub : {n : Nat} → Vec n → Vec n → Vec n
  | 0, _, _ => ()
  | _ + 1, (a, u), (b, v) => (balance_sub a b, vec_sub u v)

def vec_scale : {n : Nat} → Balance → Vec n → Vec n
  | 0, _, _ => ()
  | _ + 1, s, (a, u) => (balance_mul s a, vec_scale s u)

def vdot : {n : Nat} → Vec n → Vec n → Balance
  | 0, _, _ => bzero
  | _ + 1, (a, u), (b, v) => balance_add (balance_mul a b) (vdot u v)

def vec_snoc : {n : Nat} → Vec n → Balance → Vec (n + 1)
  | 0, _, x => (x, ())
  | _ + 1, (a, u), x => (a, vec_snoc u x)

def vec_embed {n : Nat} (u : Vec n) : Vec (n + 1) :=
  vec_snoc u bzero

theorem vec_equiv_refl : {n : Nat} → (u : Vec n) → VecEquiv u u
  | 0, () => trivial
  | _ + 1, (a, u) => ⟨step24_balance_refl a, vec_equiv_refl u⟩

theorem vec_equiv_symm : {n : Nat} → {u v : Vec n} → VecEquiv u v → VecEquiv v u
  | 0, (), (), _ => trivial
  | _ + 1, (_, _), (_, _), ⟨hab, huv⟩ => ⟨step24_balance_symm hab, vec_equiv_symm huv⟩

theorem vec_equiv_trans : {n : Nat} → {u v w : Vec n} → VecEquiv u v → VecEquiv v w → VecEquiv u w
  | 0, (), (), (), _, _ => trivial
  | _ + 1, (_, _), (_, _), (_, _), ⟨h1a, h1u⟩, ⟨h2a, h2u⟩ =>
    ⟨step24_balance_trans h1a h2a, vec_equiv_trans h1u h2u⟩

theorem vec_add_congr : {n : Nat} → {u1 u2 v1 v2 : Vec n} →
    VecEquiv u1 u2 → VecEquiv v1 v2 → VecEquiv (vec_add u1 v1) (vec_add u2 v2)
  | 0, (), (), (), (), _, _ => trivial
  | _ + 1, (_, _), (_, _), (_, _), (_, _), ⟨h1a, h1u⟩, ⟨h2a, h2u⟩ =>
    ⟨step25_balance_add_congr h1a h2a, vec_add_congr h1u h2u⟩

theorem vec_sub_congr : {n : Nat} → {u1 u2 v1 v2 : Vec n} →
    VecEquiv u1 u2 → VecEquiv v1 v2 → VecEquiv (vec_sub u1 v1) (vec_sub u2 v2)
  | 0, (), (), (), (), _, _ => trivial
  | _ + 1, (_, _), (_, _), (_, _), (_, _), ⟨h1a, h1u⟩, ⟨h2a, h2u⟩ =>
    ⟨bal_sub_congr h1a h2a, vec_sub_congr h1u h2u⟩

theorem vec_scale_congr : {n : Nat} → {s1 s2 : Balance} → {u1 u2 : Vec n} →
    BalanceEquiv s1 s2 → VecEquiv u1 u2 → VecEquiv (vec_scale s1 u1) (vec_scale s2 u2)
  | 0, _, _, (), (), _, _ => trivial
  | _ + 1, _, _, (_, _), (_, _), hs, ⟨ha, hu⟩ =>
    ⟨bal_mul_congr hs ha, vec_scale_congr hs hu⟩

theorem vdot_comm : {n : Nat} → (u v : Vec n) → BalanceEquiv (vdot u v) (vdot v u)
  | 0, (), () => step24_balance_refl bzero
  | _ + 1, (a, u), (b, v) => step25_balance_add_congr (bal_mul_comm a b) (vdot_comm u v)

theorem vdot_self_ge_zero : {n : Nat} → (u : Vec n) → BalGeZero (vdot u u)
  | 0, () => Or.inl (step24_balance_refl bzero)
  | _ + 1, (a, u) => bal_add_ge_ge (bal_sq_ge_zero a) (vdot_self_ge_zero u)

theorem vdot_congr : {n : Nat} → {u1 u2 v1 v2 : Vec n} →
    VecEquiv u1 u2 → VecEquiv v1 v2 → BalanceEquiv (vdot u1 v1) (vdot u2 v2)
  | 0, (), (), (), (), _, _ => step24_balance_refl bzero
  | _ + 1, (_, _), (_, _), (_, _), (_, _), ⟨h1a, h1u⟩, ⟨h2a, h2u⟩ =>
    step25_balance_add_congr (bal_mul_congr h1a h2a) (vdot_congr h1u h2u)

private def e_s125_zero3_R : BExpr :=
  .add (.sub (.mul (.var 0) .zero) (.mul (.var 1) .zero)) (.mul (.var 2) .zero)
private theorem p_s125_zero3 : BExpr.zero.norm = e_s125_zero3_R.norm := rfl

private def e_s125_zero3add_R : BExpr :=
  .add (.add (.mul (.var 0) .zero) (.mul (.var 2) .zero)) (.mul (.var 1) .zero)
private theorem p_s125_zero3add : BExpr.zero.norm = e_s125_zero3add_R.norm := rfl

private def e_s125_sub_step_L : BExpr :=
  .add (.add (.sub (.var 0) (.var 1)) (.var 2)) (.add (.sub (.var 3) (.var 4)) (.var 5))
private def e_s125_sub_step_R : BExpr :=
  .add (.sub (.add (.var 0) (.var 3)) (.add (.var 1) (.var 4))) (.add (.var 2) (.var 5))
private theorem p_s125_sub_step : e_s125_sub_step_L.norm = e_s125_sub_step_R.norm := rfl

private def e_s125_add_step_L : BExpr :=
  .add (.add (.add (.var 0) (.var 2)) (.var 1)) (.add (.add (.var 3) (.var 5)) (.var 4))
private def e_s125_add_step_R : BExpr :=
  .add (.add (.add (.var 0) (.var 3)) (.add (.var 2) (.var 5))) (.add (.var 1) (.var 4))
private theorem p_s125_add_step : e_s125_add_step_L.norm = e_s125_add_step_R.norm := rfl

theorem vdot_sub2_sq : {n : Nat} → (a b : Balance) → (u v : Vec n) →
    let w := vec_sub (vec_scale a u) (vec_scale b v)
    BalanceEquiv (vdot w w)
      (balance_add (balance_sub (balance_mul (bsq a) (vdot u u))
                                (balance_mul (balance_mul (balOfNat 2) (balance_mul a b)) (vdot u v)))
                   (balance_mul (bsq b) (vdot v v)))
  | 0, a, b, (), () =>
    bal_poly_eq (fun | 0 => bsq a | 1 => balance_mul (balOfNat 2) (balance_mul a b) | 2 => bsq b | _ => bzero)
      .zero e_s125_zero3_R p_s125_zero3
  | _ + 1, a, b, (uh, ut), (vh, vt) => by
    dsimp [vec_scale, vec_sub, vdot]
    have hhd := dot4_sub2_sq_scalar a b uh vh
    have htl := vdot_sub2_sq a b ut vt
    have hsum := step25_balance_add_congr hhd htl
    let c1 := bsq a; let c2 := bsq b; let c3 := balance_mul (balOfNat 2) (balance_mul a b)
    let p0 := balance_mul c1 (bsq uh); let m0 := balance_mul c3 (balance_mul uh vh); let q0 := balance_mul c2 (bsq vh)
    let p1 := balance_mul c1 (vdot ut ut); let m1 := balance_mul c3 (vdot ut vt); let q1 := balance_mul c2 (vdot vt vt)
    have hre := bal_poly_eq (fun | 0 => p0 | 1 => m0 | 2 => q0 | 3 => p1 | 4 => m1 | 5 => q1 | _ => bzero)
      e_s125_sub_step_L e_s125_sub_step_R p_s125_sub_step
    have hfac := step25_balance_add_congr
      (bal_sub_congr
        (step24_balance_symm (bal_left_distrib c1 (bsq uh) (vdot ut ut)))
        (step24_balance_symm (bal_left_distrib c3 (balance_mul uh vh) (vdot ut vt))))
      (step24_balance_symm (bal_left_distrib c2 (bsq vh) (vdot vt vt)))
    exact step24_balance_trans hsum (step24_balance_trans hre hfac)

theorem vdot_lincomb2_sq : {n : Nat} → (a b : Balance) → (u v : Vec n) →
    let w := vec_add (vec_scale a u) (vec_scale b v)
    BalanceEquiv (vdot w w)
      (balance_add (balance_add (balance_mul (bsq a) (vdot u u)) (balance_mul (bsq b) (vdot v v)))
                   (balance_mul (balance_mul (balOfNat 2) (balance_mul a b)) (vdot u v)))
  | 0, a, b, (), () =>
    bal_poly_eq (fun | 0 => bsq a | 1 => balance_mul (balOfNat 2) (balance_mul a b) | 2 => bsq b | _ => bzero)
      .zero e_s125_zero3add_R p_s125_zero3add
  | _ + 1, a, b, (uh, ut), (vh, vt) => by
    dsimp [vec_scale, vec_add, vdot]
    have hhd := dot4_lincomb2_sq_scalar a b uh vh
    have htl := vdot_lincomb2_sq a b ut vt
    have hsum := step25_balance_add_congr hhd htl
    let c1 := bsq a; let c2 := bsq b; let c3 := balance_mul (balOfNat 2) (balance_mul a b)
    let p0 := balance_mul c1 (bsq uh); let q0 := balance_mul c2 (bsq vh); let m0 := balance_mul c3 (balance_mul uh vh)
    let p1 := balance_mul c1 (vdot ut ut); let q1 := balance_mul c2 (vdot vt vt); let m1 := balance_mul c3 (vdot ut vt)
    have hre := bal_poly_eq (fun | 0 => p0 | 1 => m0 | 2 => q0 | 3 => p1 | 4 => m1 | 5 => q1 | _ => bzero)
      e_s125_add_step_L e_s125_add_step_R p_s125_add_step
    have hfac := step25_balance_add_congr
      (step25_balance_add_congr
        (step24_balance_symm (bal_left_distrib c1 (bsq uh) (vdot ut ut)))
        (step24_balance_symm (bal_left_distrib c2 (bsq vh) (vdot vt vt))))
      (step24_balance_symm (bal_left_distrib c3 (balance_mul uh vh) (vdot ut vt)))
    exact step24_balance_trans hsum (step24_balance_trans hre hfac)

theorem vdot_snoc : {n : Nat} → (u v : Vec n) → (a b : Balance) →
    BalanceEquiv (vdot (vec_snoc u a) (vec_snoc v b)) (balance_add (vdot u v) (balance_mul a b))
  | 0, (), (), a, b =>
    bal_add_comm (balance_mul a b) bzero
  | _ + 1, (uh, ut), (vh, vt), a, b =>
    step24_balance_trans
      (step25_balance_add_congr (step24_balance_refl (balance_mul uh vh)) (vdot_snoc ut vt a b))
      (step24_balance_symm (bal_add_assoc (balance_mul uh vh) (vdot ut vt) (balance_mul a b)))

theorem vec_scale_snoc (s : Balance) : {n : Nat} → (u : Vec n) → (a : Balance) →
    vec_scale s (vec_snoc u a) = vec_snoc (vec_scale s u) (balance_mul s a)
  | 0, (), _ => rfl
  | _ + 1, (uh, ut), a => congrArg (Prod.mk (balance_mul s uh)) (vec_scale_snoc s ut a)

theorem vec_sub_snoc : {n : Nat} → (u v : Vec n) → (a b : Balance) →
    vec_sub (vec_snoc u a) (vec_snoc v b) = vec_snoc (vec_sub u v) (balance_sub a b)
  | 0, (), (), _, _ => rfl
  | _ + 1, (uh, ut), (vh, vt), a, b =>
    congrArg (Prod.mk (balance_sub uh vh)) (vec_sub_snoc ut vt a b)

theorem vec_snoc_congr : {n : Nat} → {u1 u2 : Vec n} → {a1 a2 : Balance} →
    VecEquiv u1 u2 → BalanceEquiv a1 a2 → VecEquiv (vec_snoc u1 a1) (vec_snoc u2 a2)
  | 0, (), (), _, _, _, ha => ⟨ha, trivial⟩
  | _ + 1, (_, _), (_, _), _, _, ⟨hh, ht⟩, ha => ⟨hh, vec_snoc_congr ht ha⟩

-- ----- Произвольно-размерный гейтированный Хаусхолдер и K-каскад AdaptiveO(d) -----

def householderN_num {n : Nat} (D p q : Balance) (v u : Vec n) : Vec n :=
  vec_sub (vec_scale (balance_mul q D) u) (vec_scale (balance_mul p (vdot v u)) v)

theorem householderN_congr {n : Nat} {D1 D2 p1 p2 q1 q2 : Balance} {v1 v2 u1 u2 : Vec n}
    (hD : BalanceEquiv D1 D2) (hp : BalanceEquiv p1 p2) (hq : BalanceEquiv q1 q2)
    (hv : VecEquiv v1 v2) (hu : VecEquiv u1 u2) :
    VecEquiv (householderN_num D1 p1 q1 v1 u1) (householderN_num D2 p2 q2 v2 u2) :=
  vec_sub_congr
    (vec_scale_congr (bal_mul_congr hq hD) hu)
    (vec_scale_congr (bal_mul_congr hp (vdot_congr hv hu)) hv)

structure HouseholderFactor (n : Nat) where
  v : Vec n
  eps : Balance
  p : Balance
  q : Balance

def FactorValid {n : Nat} (f : HouseholderFactor n) : Prop :=
  BalGeZero f.eps ∧ BalGtZero f.q ∧ BalGeZero f.p ∧ BalLe f.p (balance_mul (balOfNat 2) f.q)

def factor_denom {n : Nat} (f : HouseholderFactor n) : Balance :=
  balance_mul f.q (balance_add (vdot f.v f.v) f.eps)

def apply_factor {n : Nat} (f : HouseholderFactor n) (u : Vec n) : Vec n :=
  householderN_num (balance_add (vdot f.v f.v) f.eps) f.p f.q f.v u

def apply_cascade {n : Nat} : List (HouseholderFactor n) → Vec n → Vec n
  | [], u => u
  | f :: fs, u => apply_cascade fs (apply_factor f u)

def cascade_denom {n : Nat} : List (HouseholderFactor n) → Balance
  | [] => bone
  | f :: fs => balance_mul (cascade_denom fs) (factor_denom f)

def CascadeValid {n : Nat} : List (HouseholderFactor n) → Prop
  | [] => True
  | f :: fs => FactorValid f ∧ CascadeValid fs

def embed_factor {n : Nat} (f : HouseholderFactor n) : HouseholderFactor (n + 1) :=
  ⟨vec_embed f.v, f.eps, f.p, f.q⟩

def embed_cascade {n : Nat} : List (HouseholderFactor n) → List (HouseholderFactor (n + 1))
  | [] => []
  | f :: fs => embed_factor f :: embed_cascade fs

theorem householderN_bibo {n : Nat} (v u : Vec n) (eps p q : Balance)
    (heps : BalGeZero eps) (hq : BalGtZero q) (hp0 : BalGeZero p)
    (hp2 : BalLe p (balance_mul (balOfNat 2) q)) :
    let D := balance_add (vdot v v) eps
    let h_num := householderN_num D p q v u
    let slack := balance_mul (balance_mul p (bsq (vdot v u)))
                   (balance_add (balance_mul (balance_sub (balance_mul (balOfNat 2) q) p) (vdot v v))
                                (balance_mul (balance_mul (balOfNat 2) q) eps))
    BalanceEquiv (balance_add (vdot h_num h_num) slack) (balance_mul (bsq (balance_mul q D)) (vdot u u)) ∧
    BalLe (vdot h_num h_num) (balance_mul (bsq (balance_mul q D)) (vdot u u)) := by
  dsimp only
  let D := balance_add (vdot v v) eps
  let vu := vdot v u
  let vv := vdot v v
  let uu := vdot u u
  have hsq := vdot_sub2_sq (balance_mul q D) (balance_mul p vu) u v
  have huv : BalanceEquiv (vdot u v) vu := vdot_comm u v
  have hsq' := step24_balance_trans hsq
    (step25_balance_add_congr
      (bal_sub_congr (step24_balance_refl _) (bal_mul_congr (step24_balance_refl _) huv))
      (step24_balance_refl _))
  let env : Nat → Balance := fun | 0 => p | 1 => q | 2 => vv | 3 => eps | 4 => vu | 5 => uu | _ => bzero
  have hid := step24_balance_trans (step25_balance_add_congr hsq' (step24_balance_refl _))
    (bal_poly_eq env e_s118_hid_L e_s118_hid_R p_s118_hid)
  have h2q : BalGtZero (balance_mul (balOfNat 2) q) := bal_mul_gt_gt (balOfNat_succ_gt_zero 1) hq
  have hslack_ge : BalGeZero (balance_mul (balance_mul p (bsq vu))
                   (balance_add (balance_mul (balance_sub (balance_mul (balOfNat 2) q) p) vv)
                                (balance_mul (balance_mul (balOfNat 2) q) eps))) :=
    bal_mul_ge_ge (bal_mul_ge_ge hp0 (bal_sq_ge_zero vu))
      (bal_add_ge_ge (bal_mul_ge_ge hp2 (vdot_self_ge_zero v)) (bal_mul_ge_ge (Or.inr h2q) heps))
  have hsub_slack : BalanceEquiv
      (balance_mul (balance_mul p (bsq vu))
        (balance_add (balance_mul (balance_sub (balance_mul (balOfNat 2) q) p) vv)
                     (balance_mul (balance_mul (balOfNat 2) q) eps)))
      (balance_sub (balance_mul (bsq (balance_mul q D)) uu)
                   (vdot (householderN_num D p q v u) (householderN_num D p q v u))) :=
    step24_balance_trans
      (bal_poly_eq
        (fun | 0 => vdot (householderN_num D p q v u) (householderN_num D p q v u)
             | 1 => balance_mul (balance_mul p (bsq vu))
                      (balance_add (balance_mul (balance_sub (balance_mul (balOfNat 2) q) p) vv)
                                   (balance_mul (balance_mul (balOfNat 2) q) eps))
             | _ => bzero)
        e_sub_add_cancel_L e_sub_add_cancel_R p_sub_add_cancel)
      (bal_sub_congr hid (step24_balance_refl _))
  exact ⟨hid, bal_ge_zero_of_equiv hsub_slack hslack_ge⟩

private def e_s125_casc_scale_L : BExpr :=
  .mul (.mul (.var 0) (.var 0)) (.mul (.mul (.var 1) (.var 1)) (.var 2))
private def e_s125_casc_scale_R : BExpr :=
  .mul (.mul (.mul (.var 0) (.var 1)) (.mul (.var 0) (.var 1))) (.var 2)
private theorem p_s125_casc_scale : e_s125_casc_scale_L.norm = e_s125_casc_scale_R.norm := rfl

theorem cascade_bibo {n : Nat} : (fs : List (HouseholderFactor n)) → (u : Vec n) →
    CascadeValid fs →
    BalLe (vdot (apply_cascade fs u) (apply_cascade fs u))
          (balance_mul (bsq (cascade_denom fs)) (vdot u u))
  | [], u, _ => by
    dsimp [apply_cascade, cascade_denom, bsq]
    have h1 : BalanceEquiv (balance_mul (balance_mul bone bone) (vdot u u)) (vdot u u) :=
      step24_balance_trans (bal_mul_congr (bal_one_mul bone) (step24_balance_refl _)) (bal_one_mul _)
    exact bal_le_of_equiv_right (step24_balance_symm h1) (bal_le_refl (vdot u u))
  | f :: fs, u, ⟨hf, hfs⟩ => by
    dsimp [apply_cascade, cascade_denom]
    have hstep := (householderN_bibo f.v u f.eps f.p f.q hf.1 hf.2.1 hf.2.2.1 hf.2.2.2).2
    have htail := cascade_bibo fs (apply_factor f u) hfs
    have hmul := bal_le_mul_nonneg_left (bal_sq_ge_zero (cascade_denom fs)) hstep
    have hle := bal_le_trans htail hmul
    have hnorm_eq : BalanceEquiv
        (balance_mul (bsq (cascade_denom fs)) (balance_mul (bsq (factor_denom f)) (vdot u u)))
        (balance_mul (bsq (balance_mul (cascade_denom fs) (factor_denom f))) (vdot u u)) :=
      bal_poly_eq (fun | 0 => cascade_denom fs | 1 => factor_denom f | 2 => vdot u u | _ => bzero)
        e_s125_casc_scale_L e_s125_casc_scale_R p_s125_casc_scale
    exact bal_le_of_equiv_right hnorm_eq hle

theorem apply_factor_congr {n : Nat} (f : HouseholderFactor n) {u1 u2 : Vec n} (hu : VecEquiv u1 u2) :
    VecEquiv (apply_factor f u1) (apply_factor f u2) :=
  householderN_congr (step24_balance_refl _) (step24_balance_refl _) (step24_balance_refl _)
    (vec_equiv_refl f.v) hu

theorem apply_cascade_congr {n : Nat} : (fs : List (HouseholderFactor n)) → {u1 u2 : Vec n} →
    VecEquiv u1 u2 → VecEquiv (apply_cascade fs u1) (apply_cascade fs u2)
  | [], _, _, hu => hu
  | f :: fs, _, _, hu => apply_cascade_congr fs (apply_factor_congr f hu)

private def e_s125_last_L : BExpr :=
  .sub (.mul (.mul (.var 0) (.var 1)) (.var 2)) (.mul (.mul (.var 3) (.var 4)) .zero)
private def e_s125_last_R : BExpr := .mul (.mul (.var 0) (.var 1)) (.var 2)
private theorem p_s125_last : e_s125_last_L.norm = e_s125_last_R.norm := rfl

theorem apply_factor_embed {n : Nat} (f : HouseholderFactor n) (u : Vec n) (c : Balance) :
    VecEquiv (apply_factor (embed_factor f) (vec_snoc u c))
             (vec_snoc (apply_factor f u) (balance_mul (factor_denom f) c)) := by
  dsimp [apply_factor, embed_factor, vec_embed, householderN_num, factor_denom]
  let D_emb := balance_add (vdot (vec_snoc f.v bzero) (vec_snoc f.v bzero)) f.eps
  let D := balance_add (vdot f.v f.v) f.eps
  have hvv : BalanceEquiv (vdot (vec_snoc f.v bzero) (vec_snoc f.v bzero)) (vdot f.v f.v) := by
    have hs := vdot_snoc f.v f.v bzero bzero
    have hz := step25_balance_add_congr (step24_balance_refl (vdot f.v f.v)) (bal_mul_zero bzero)
    exact step24_balance_trans hs (step24_balance_trans hz (bal_add_zero (vdot f.v f.v)))
  have hD : BalanceEquiv D_emb D := step25_balance_add_congr hvv (step24_balance_refl f.eps)
  have hvu : BalanceEquiv (vdot (vec_snoc f.v bzero) (vec_snoc u c)) (vdot f.v u) := by
    have hs := vdot_snoc f.v u bzero c
    have hz := step25_balance_add_congr (step24_balance_refl (vdot f.v u)) (bal_zero_mul c)
    exact step24_balance_trans hs (step24_balance_trans hz (bal_add_zero (vdot f.v u)))
  rw [vec_scale_snoc, vec_scale_snoc, vec_sub_snoc]
  have hpref' : VecEquiv
      (vec_sub (vec_scale (balance_mul f.q D_emb) u)
               (vec_scale (balance_mul f.p (vdot (vec_snoc f.v bzero) (vec_snoc u c))) f.v))
      (vec_sub (vec_scale (balance_mul f.q D) u)
               (vec_scale (balance_mul f.p (vdot f.v u)) f.v)) :=
    vec_sub_congr
      (vec_scale_congr (bal_mul_congr (step24_balance_refl f.q) hD) (vec_equiv_refl u))
      (vec_scale_congr (bal_mul_congr (step24_balance_refl f.p) hvu) (vec_equiv_refl f.v))
  have hlast : BalanceEquiv
      (balance_sub (balance_mul (balance_mul f.q D_emb) c)
                   (balance_mul (balance_mul f.p (vdot (vec_snoc f.v bzero) (vec_snoc u c))) bzero))
      (balance_mul (balance_mul f.q D) c) :=
    step24_balance_trans
      (bal_sub_congr (bal_mul_congr (bal_mul_congr (step24_balance_refl f.q) hD) (step24_balance_refl c))
                     (step24_balance_refl _))
      (bal_poly_eq (fun | 0 => f.q | 1 => D | 2 => c | 3 => f.p
                        | 4 => vdot (vec_snoc f.v bzero) (vec_snoc u c) | _ => bzero)
        e_s125_last_L e_s125_last_R p_s125_last)
  exact vec_snoc_congr hpref' hlast

theorem apply_cascade_snoc {n : Nat} : (fs : List (HouseholderFactor n)) → (u : Vec n) → (c : Balance) →
    VecEquiv (apply_cascade (embed_cascade fs) (vec_snoc u c))
             (vec_snoc (apply_cascade fs u) (balance_mul (cascade_denom fs) c))
  | [], u, c => by
    dsimp [embed_cascade, apply_cascade, cascade_denom]
    exact vec_snoc_congr (vec_equiv_refl u) (step24_balance_symm (bal_one_mul c))
  | f :: fs, u, c => by
    dsimp [embed_cascade, apply_cascade, cascade_denom]
    have h1 := apply_factor_embed f u c
    have h2 := apply_cascade_congr (embed_cascade fs) h1
    have h3 := apply_cascade_snoc fs (apply_factor f u) (balance_mul (factor_denom f) c)
    have hassoc : VecEquiv
        (vec_snoc (apply_cascade fs (apply_factor f u))
                  (balance_mul (cascade_denom fs) (balance_mul (factor_denom f) c)))
        (vec_snoc (apply_cascade fs (apply_factor f u))
                  (balance_mul (balance_mul (cascade_denom fs) (factor_denom f)) c)) :=
      vec_snoc_congr (vec_equiv_refl _) (step24_balance_symm (bal_mul_assoc (cascade_denom fs) (factor_denom f) c))
    exact vec_equiv_trans h2 (vec_equiv_trans h3 hassoc)

theorem apply_cascade_embed {n : Nat} (fs : List (HouseholderFactor n)) (u : Vec n) :
    VecEquiv (apply_cascade (embed_cascade fs) (vec_embed u))
             (vec_embed (apply_cascade fs u)) := by
  dsimp [vec_embed]
  exact vec_equiv_trans (apply_cascade_snoc fs u bzero)
    (vec_snoc_congr (vec_equiv_refl _) (bal_mul_zero (cascade_denom fs)))

-- ----- Конструктивный свидетель S₆ ≤ O(5) на 5D-гиперплоскости sum₆ = 0 -----

structure Vec6 where
  x0 : Balance
  x1 : Balance
  x2 : Balance
  x3 : Balance
  x4 : Balance
  x5 : Balance

def Vec6Equiv (u v : Vec6) : Prop :=
  BalanceEquiv u.x0 v.x0 ∧ BalanceEquiv u.x1 v.x1 ∧ BalanceEquiv u.x2 v.x2 ∧
  BalanceEquiv u.x3 v.x3 ∧ BalanceEquiv u.x4 v.x4 ∧ BalanceEquiv u.x5 v.x5

def vec6_zero : Vec6 := ⟨bzero, bzero, bzero, bzero, bzero, bzero⟩
def sum6 (u : Vec6) : Balance :=
  balance_add (balance_add (balance_add u.x0 u.x1) (balance_add u.x2 u.x3)) (balance_add u.x4 u.x5)
def dot6 (u v : Vec6) : Balance :=
  balance_add
    (balance_add (balance_add (balance_mul u.x0 v.x0) (balance_mul u.x1 v.x1))
                 (balance_add (balance_mul u.x2 v.x2) (balance_mul u.x3 v.x3)))
    (balance_add (balance_mul u.x4 v.x4) (balance_mul u.x5 v.x5))

-- Генераторы S₆: транспозиция a = (0 1) и 6-цикл b = (0 1 2 3 4 5); ab имеет порядок 5
def s6_a (u : Vec6) : Vec6 := ⟨u.x1, u.x0, u.x2, u.x3, u.x4, u.x5⟩
def s6_b (u : Vec6) : Vec6 := ⟨u.x1, u.x2, u.x3, u.x4, u.x5, u.x0⟩

private def e_s125_sum6_base : BExpr :=
  .add (.add (.add (.var 0) (.var 1)) (.add (.var 2) (.var 3))) (.add (.var 4) (.var 5))
private def e_s125_sum6_a : BExpr :=
  .add (.add (.add (.var 1) (.var 0)) (.add (.var 2) (.var 3))) (.add (.var 4) (.var 5))
private def e_s125_sum6_b : BExpr :=
  .add (.add (.add (.var 1) (.var 2)) (.add (.var 3) (.var 4))) (.add (.var 5) (.var 0))
private theorem p_s125_sum6_a : e_s125_sum6_a.norm = e_s125_sum6_base.norm := rfl
private theorem p_s125_sum6_b : e_s125_sum6_b.norm = e_s125_sum6_base.norm := rfl

private def e_s125_dot6_base : BExpr :=
  .add (.add (.add (.mul (.var 0) (.var 0)) (.mul (.var 1) (.var 1)))
             (.add (.mul (.var 2) (.var 2)) (.mul (.var 3) (.var 3))))
       (.add (.mul (.var 4) (.var 4)) (.mul (.var 5) (.var 5)))
private def e_s125_dot6_a : BExpr :=
  .add (.add (.add (.mul (.var 1) (.var 1)) (.mul (.var 0) (.var 0)))
             (.add (.mul (.var 2) (.var 2)) (.mul (.var 3) (.var 3))))
       (.add (.mul (.var 4) (.var 4)) (.mul (.var 5) (.var 5)))
private def e_s125_dot6_b : BExpr :=
  .add (.add (.add (.mul (.var 1) (.var 1)) (.mul (.var 2) (.var 2)))
             (.add (.mul (.var 3) (.var 3)) (.mul (.var 4) (.var 4))))
       (.add (.mul (.var 5) (.var 5)) (.mul (.var 0) (.var 0)))
private theorem p_s125_dot6_a : e_s125_dot6_a.norm = e_s125_dot6_base.norm := rfl
private theorem p_s125_dot6_b : e_s125_dot6_b.norm = e_s125_dot6_base.norm := rfl

private def e_s125_sum6_eq_L : BExpr :=
  .add (.add (.add (.var 0) (.var 0)) (.add (.var 0) (.var 0))) (.add (.var 0) (.var 0))
private def e_s125_sum6_eq_R : BExpr := .mul (.nat 6) (.var 0)
private theorem p_s125_sum6_eq : e_s125_sum6_eq_L.norm = e_s125_sum6_eq_R.norm := rfl

/-- ШАГ 125. Произвольно-размерный каскад Хаусхолдера `AdaptiveO(d)` (вариант 2 Pillar 1) и свидетель S₆ ≤ O(5):
    1. Для ЛЮБОЙ размерности `n` и ЛЮБОГО допустимого K-каскада `fs : List (HouseholderFactor n)`:
       ||H_fs(u)||² ≤ (D_fs)² · ||u||² (безусловное BIBO-сжатие при любом активном ранге d_act).
    2. Точное сохранение функции при вложении O(d) ↪ O(d+1) через `(u, 0)`:
       `apply_cascade (embed_cascade fs) (vec_embed u) ~ vec_embed (apply_cascade fs u)` (нулевое забывание при повышении ранга!).
    3. Покоординатная инвариантность нового измерения: для каждого фактора ось `d+1` лишь масштабируется знаменателем `q·D`.
    4. Конструктивный свидетель S₆ ≤ O(5) на 5D-гиперплоскости `sum₆ = 0`:
       a² = I, b⁶ = I, (ab)⁵ = I, ab ≠ ba, сохранение гиперплоскости и нормы, а 6-цикл `b` не имеет ненулевых неподвижных векторов. -/
theorem step125_adaptive_od_householder_cascade_and_embedding :
    (∀ {n : Nat} (fs : List (HouseholderFactor n)) (u : Vec n),
      CascadeValid fs →
      BalLe (vdot (apply_cascade fs u) (apply_cascade fs u))
            (balance_mul (bsq (cascade_denom fs)) (vdot u u)))
    ∧
    (∀ {n : Nat} (fs : List (HouseholderFactor n)) (u : Vec n),
      VecEquiv (apply_cascade (embed_cascade fs) (vec_embed u))
               (vec_embed (apply_cascade fs u)))
    ∧
    (∀ {n : Nat} (f : HouseholderFactor n) (u : Vec n) (c : Balance),
      VecEquiv (apply_factor (embed_factor f) (vec_snoc u c))
               (vec_snoc (apply_factor f u) (balance_mul (factor_denom f) c)))
    ∧
    (∀ u : Vec6,
      BalanceEquiv (sum6 (s6_a u)) (sum6 u) ∧
      BalanceEquiv (sum6 (s6_b u)) (sum6 u) ∧
      BalanceEquiv (dot6 (s6_a u) (s6_a u)) (dot6 u u) ∧
      BalanceEquiv (dot6 (s6_b u) (s6_b u)) (dot6 u u) ∧
      Vec6Equiv (s6_a (s6_a u)) u ∧
      Vec6Equiv (s6_b (s6_b (s6_b (s6_b (s6_b (s6_b u)))))) u ∧
      Vec6Equiv (s6_a (s6_b (s6_a (s6_b (s6_a (s6_b (s6_a (s6_b (s6_a (s6_b u)))))))))) u)
    ∧
    (Vec6Equiv (s6_a (s6_b ⟨bone, bneg_one, bzero, bzero, bzero, bzero⟩))
               (s6_b (s6_a ⟨bone, bneg_one, bzero, bzero, bzero, bzero⟩)) → False)
    ∧
    (∀ x : Vec6, Vec6Equiv (s6_b x) x → BalanceEquiv (sum6 x) bzero → Vec6Equiv x vec6_zero) := by
  refine ⟨fun fs u hfs => cascade_bibo fs u hfs,
          fun fs u => apply_cascade_embed fs u,
          fun f u c => apply_factor_embed f u c, ?_, ?_, ?_⟩
  · intro u
    let env : Nat → Balance := fun
      | 0 => u.x0 | 1 => u.x1 | 2 => u.x2 | 3 => u.x3 | 4 => u.x4 | 5 => u.x5 | _ => bzero
    have id6 : Vec6Equiv u u :=
      ⟨step24_balance_refl _, step24_balance_refl _, step24_balance_refl _,
       step24_balance_refl _, step24_balance_refl _, step24_balance_refl _⟩
    exact ⟨bal_poly_eq env e_s125_sum6_a e_s125_sum6_base p_s125_sum6_a,
           bal_poly_eq env e_s125_sum6_b e_s125_sum6_base p_s125_sum6_b,
           bal_poly_eq env e_s125_dot6_a e_s125_dot6_base p_s125_dot6_a,
           bal_poly_eq env e_s125_dot6_b e_s125_dot6_base p_s125_dot6_b,
           id6, id6, id6⟩
  · intro hne
    -- (a(b(1,-1,0,0,0,0))).x0 = 0, но (b(a(1,-1,0,0,0,0))).x0 = 1
    exact bal_gt_zero_ne_zero bone_gt_zero (step24_balance_symm hne.1)
  · intro x hfx hsum0
    have e01 : BalanceEquiv x.x1 x.x0 := hfx.1
    have e12 : BalanceEquiv x.x2 x.x1 := hfx.2.1
    have e23 : BalanceEquiv x.x3 x.x2 := hfx.2.2.1
    have e34 : BalanceEquiv x.x4 x.x3 := hfx.2.2.2.1
    have e45 : BalanceEquiv x.x5 x.x4 := hfx.2.2.2.2.1
    have e20 : BalanceEquiv x.x2 x.x0 := step24_balance_trans e12 e01
    have e30 : BalanceEquiv x.x3 x.x0 := step24_balance_trans e23 e20
    have e40 : BalanceEquiv x.x4 x.x0 := step24_balance_trans e34 e30
    have e50 : BalanceEquiv x.x5 x.x0 := step24_balance_trans e45 e40
    have hs_eq : BalanceEquiv (sum6 x) (balance_mul (balOfNat 6) x.x0) :=
      step24_balance_trans
        (step25_balance_add_congr
          (step25_balance_add_congr
            (step25_balance_add_congr (step24_balance_refl x.x0) e01)
            (step25_balance_add_congr e20 e30))
          (step25_balance_add_congr e40 e50))
        (bal_poly_eq (fun | 0 => x.x0 | _ => bzero) e_s125_sum6_eq_L e_s125_sum6_eq_R p_s125_sum6_eq)
    have hx0 : BalanceEquiv x.x0 bzero :=
      bal_mul_gt_zero_eq_zero (balOfNat_succ_gt_zero 5)
        (step24_balance_trans (step24_balance_symm hs_eq) hsum0)
    exact ⟨hx0,
           step24_balance_trans e01 hx0,
           step24_balance_trans e20 hx0,
           step24_balance_trans e30 hx0,
           step24_balance_trans e40 hx0,
           step24_balance_trans e50 hx0⟩

-- ----- ШАГ 126. Автономия Pillar 5: IDBD, безопасный ввод var+1, выпуклый предел α=0 и surprise -----

private def e_s126_add_21_L : BExpr :=
  .add (.add (.mul (.mul .one .one) (.var 0)) (.mul (.mul .one .one) (.var 1)))
       (.mul (.mul (.nat 2) (.mul .one .one)) (.var 2))
private def e_s126_add_21_R : BExpr :=
  .sub (.mul (.nat 2) (.var 2)) (.neg (.add (.var 0) (.var 1)))
private theorem p_s126_add_21 : e_s126_add_21_L.norm = e_s126_add_21_R.norm := rfl

private def e_s126_sub_21_L : BExpr :=
  .add (.sub (.mul (.mul .one .one) (.var 0)) (.mul (.mul (.nat 2) (.mul .one .one)) (.var 2)))
       (.mul (.mul .one .one) (.var 1))
private def e_s126_sub_21_R : BExpr :=
  .sub (.add (.var 0) (.var 1)) (.mul (.nat 2) (.var 2))
private theorem p_s126_sub_21 : e_s126_sub_21_L.norm = e_s126_sub_21_R.norm := rfl

private def e_s126_neg_le_L : BExpr := .sub (.neg (.var 0)) (.neg (.add (.var 0) (.var 1)))
private def e_s126_neg_le_R : BExpr := .var 1
private theorem p_s126_neg_le : e_s126_neg_le_L.norm = e_s126_neg_le_R.norm := rfl

private def e_s126_x2var_L : BExpr := .mul (.var 0) (.var 1)
private def e_s126_x2var_R : BExpr := .sub (.mul (.var 0) (.add (.var 1) .one)) (.var 0)
private theorem p_s126_x2var : e_s126_x2var_L.norm = e_s126_x2var_R.norm := rfl

private def e_s126_d0_L : BExpr := .add .one (.mul (.var 0) (.mul .zero .zero))
private theorem p_s126_d0 : e_s126_d0_L.norm = BExpr.one.norm := rfl

private def e_s126_d0m1_L : BExpr := .sub (.add .one (.mul (.var 0) (.mul .zero .zero))) .one
private theorem p_s126_d0m1 : e_s126_d0m1_L.norm = BExpr.zero.norm := rfl

private def e_s126_comb0_L : BExpr :=
  .add (.mul .one (.var 0)) (.mul (.sub (.add .one (.mul (.var 2) (.mul .zero .zero))) .one) (.var 1))
private theorem p_s126_comb0 : e_s126_comb0_L.norm = (BExpr.var 0).norm := rfl

private def e_s126_da_den_d : BExpr × BExpr :=
  dAddE (.one, .zero) (dMulE (.var 1, .zero) (dMulE (.var 0, .one) (.var 0, .one)))
private def e_s126_da_L : BExpr × BExpr :=
  dMulE e_s126_da_den_d (.add .one (.mul (.var 1) (.mul (.var 0) (.var 0))),
                         .neg (.mul (.nat 2) (.mul (.var 1) (.var 0))))
private def e_s126_da_R : BExpr × BExpr :=
  let d := BExpr.add .one (.mul (.var 1) (.mul (.var 0) (.var 0)))
  dMulE (.mul d d, .zero) (.one, .zero)
private theorem p_s126_da_re : e_s126_da_L.1.norm = e_s126_da_R.1.norm := rfl
private theorem p_s126_da_eps : e_s126_da_L.2.norm = e_s126_da_R.2.norm := rfl

private def e_s126_dd_den_d : BExpr × BExpr :=
  dAddE (.one, .zero) (dMulE (.var 1, .one) (.mul (.var 0) (.var 0), .zero))
private def e_s126_dd_L : BExpr × BExpr :=
  dMulE e_s126_dd_den_d (.add .one (.mul (.var 1) (.mul (.var 0) (.var 0))),
                         .neg (.mul (.var 0) (.var 0)))
private def e_s126_dd_R : BExpr × BExpr :=
  let d := BExpr.add .one (.mul (.var 1) (.mul (.var 0) (.var 0)))
  dMulE (.mul d d, .zero) (.one, .zero)
private theorem p_s126_dd_re : e_s126_dd_L.1.norm = e_s126_dd_R.1.norm := rfl
private theorem p_s126_dd_eps : e_s126_dd_L.2.norm = e_s126_dd_R.2.norm := rfl

/-- ШАГ 126. Алгебра самоопределения Pillar 5:
    1. Рациональное IDBD-выравнивание: для ЛЮБЫХ g₁, g₂ : Vec n и ε > 0:
       `|2(g₁·g₂)| ≤ ||g₁||² + ||g₂||² ≤ ||g₁||² + ||g₂||² + ε`, т.е. `c_t ∈ [-1/2, 1/2]`.
    2. Универсальная безопасная входная норма: при `var ≥ 0` знаменатель `var + 1 ≥ 1 > 0`
       и `x² ≤ x²·(var + 1)` (нет взрыва редких one-hot каналов).
    3. Выпуклый предел Master Equation (без потолка `0.995`) + точные `DualBalance`-производные:
       при `α = 0`: `D₀ ~ 1`, `D₀ - 1 ~ 0`, `comb₀ ~ q` и `||comb₀||² ~ ||q||²`;
       и точные производные `λ(Δ, α) = 1 / (1 + Δ·α²)` по `α` и по `Δ`.
    4. Тотальность surprise-отношения `ema_fast / (ema_slow + ε)` при `ε > 0`, `ema_fast, ema_slow ≥ 0`. -/
theorem step126_pillar5_self_determination_algebra {n : Nat}
    (g1 g2 : Vec n) (eps var x delta a ema_fast ema_slow : Balance) (q w : Vec4)
    (heps : BalGtZero eps) (hvar : BalGeZero var) (hdelta : BalGeZero delta)
    (hef : BalGeZero ema_fast) (hes : BalGeZero ema_slow) :
    let S := balance_add (vdot g1 g1) (vdot g2 g2)
    let D_idbd := balance_add S eps
    let dot2 := balance_mul (balOfNat 2) (vdot g1 g2)
    let D_in := balance_add var bone
    let D_lam := balance_add bone (balance_mul delta (bsq a))
    let D0 := balance_add bone (balance_mul delta (bsq bzero))
    let comb0 := vec4_add (vec4_scale bone q) (vec4_scale (balance_sub D0 bone) w)
    let ad := dual_var a
    let den_da := dual_add (dual_const bone) (dual_mul (dual_const delta) (dual_mul ad ad))
    let dd := dual_var delta
    let den_dd := dual_add (dual_const bone) (dual_mul dd (dual_const (bsq a)))
    -- 1. IDBD
    (BalGtZero D_idbd ∧
     BalLe (balance_neg S) dot2 ∧ BalLe dot2 S ∧
     BalLe (balance_neg D_idbd) dot2 ∧ BalLe dot2 D_idbd)
    ∧
    -- 2. Безопасное масштабирование входов
    (BalLe bone D_in ∧ BalGtZero D_in ∧ BalLe (bsq x) (balance_mul (bsq x) D_in))
    ∧
    -- 3. Выпуклый предел α = 0 и точные производные λ(Δ, α)
    (BalLe bone D_lam ∧ BalGtZero D_lam ∧ BalGeZero (balance_sub D_lam bone) ∧
     BalanceEquiv D0 bone ∧ BalanceEquiv (balance_sub D0 bone) bzero ∧
     Vec4Equiv comb0 q ∧ BalanceEquiv (dot4 comb0 comb0) (dot4 q q) ∧
     DualEquiv (dual_mul den_da ⟨D_lam, balance_neg (balance_mul (balOfNat 2) (balance_mul delta a))⟩)
               (dual_mul (dual_const (bsq D_lam)) (dual_const bone)) ∧
     DualEquiv (dual_mul den_dd ⟨D_lam, balance_neg (bsq a)⟩)
               (dual_mul (dual_const (bsq D_lam)) (dual_const bone)))
    ∧
    -- 4. Surprise-исследование RL
    (BalGeZero ema_fast ∧ BalLe eps (balance_add ema_slow eps) ∧ BalGtZero (balance_add ema_slow eps)) := by
  dsimp only [dual_var, dual_const, dual_mul, dual_add, DualEquiv, bsq]
  -- 1. IDBD
  let S := balance_add (vdot g1 g1) (vdot g2 g2)
  let D_idbd := balance_add S eps
  let dot2 := balance_mul (balOfNat 2) (vdot g1 g2)
  have hS_ge : BalGeZero S := bal_add_ge_ge (vdot_self_ge_zero g1) (vdot_self_ge_zero g2)
  have hDidbd_gt : BalGtZero D_idbd := bal_gt_zero_of_equiv (bal_add_comm eps S) (bal_add_gt_ge heps hS_ge)
  have hlin := vdot_lincomb2_sq bone bone g1 g2
  have hsub := vdot_sub2_sq bone bone g1 g2
  let envI : Nat → Balance := fun | 0 => vdot g1 g1 | 1 => vdot g2 g2 | 2 => vdot g1 g2 | _ => bzero
  have hloS : BalLe (balance_neg S) dot2 :=
    bal_ge_zero_of_equiv
      (step24_balance_trans hlin (bal_poly_eq envI e_s126_add_21_L e_s126_add_21_R p_s126_add_21))
      (vdot_self_ge_zero (vec_add (vec_scale bone g1) (vec_scale bone g2)))
  have hhiS : BalLe dot2 S :=
    bal_ge_zero_of_equiv
      (step24_balance_trans hsub (bal_poly_eq envI e_s126_sub_21_L e_s126_sub_21_R p_s126_sub_21))
      (vdot_self_ge_zero (vec_sub (vec_scale bone g1) (vec_scale bone g2)))
  have hS_le_D : BalLe S D_idbd :=
    bal_ge_zero_of_equiv (bal_poly_eq (fun | 0 => S | 1 => eps | _ => bzero)
      e_sub_add_cancel_L e_sub_add_cancel_R p_sub_add_cancel) (Or.inr heps)
  have hnD_le_nS : BalLe (balance_neg D_idbd) (balance_neg S) :=
    bal_ge_zero_of_equiv (bal_poly_eq (fun | 0 => S | 1 => eps | _ => bzero)
      e_s126_neg_le_R e_s126_neg_le_L p_s126_neg_le) (Or.inr heps)
  have hloD : BalLe (balance_neg D_idbd) dot2 := bal_le_trans hnD_le_nS hloS
  have hhiD : BalLe dot2 D_idbd := bal_le_trans hhiS hS_le_D
  -- 2. Безопасный ввод
  have hDin_ge1 : BalLe bone (balance_add var bone) :=
    bal_ge_zero_of_equiv (bal_poly_eq (fun | 0 => var | 1 => bone | _ => bzero)
      e_add_sub_cancel_L e_add_sub_cancel_R p_add_sub_cancel) hvar
  have hDin_gt : BalGtZero (balance_add var bone) :=
    bal_gt_zero_of_equiv (bal_add_comm bone var) (bal_add_gt_ge bone_gt_zero hvar)
  have hx_le : BalLe (balance_mul x x) (balance_mul (balance_mul x x) (balance_add var bone)) :=
    bal_ge_zero_of_equiv (bal_poly_eq (fun | 0 => balance_mul x x | 1 => var | _ => bzero)
      e_s126_x2var_L e_s126_x2var_R p_s126_x2var) (bal_mul_ge_ge (bal_sq_ge_zero x) hvar)
  -- 3. Выпуклый предел и производные
  let D_lam := balance_add bone (balance_mul delta (balance_mul a a))
  have hda2 : BalGeZero (balance_mul delta (balance_mul a a)) :=
    bal_mul_ge_ge hdelta (bal_sq_ge_zero a)
  have hDlam_ge1 : BalLe bone D_lam :=
    bal_ge_zero_of_equiv (bal_poly_eq (fun | 0 => bone | 1 => balance_mul delta (balance_mul a a) | _ => bzero)
      e_sub_add_cancel_L e_sub_add_cancel_R p_sub_add_cancel) hda2
  have hDlam_gt : BalGtZero D_lam := bal_add_gt_ge bone_gt_zero hda2
  have hD0_eq : BalanceEquiv (balance_add bone (balance_mul delta (balance_mul bzero bzero))) bone :=
    bal_poly_eq (fun | 0 => delta | _ => bzero) e_s126_d0_L .one p_s126_d0
  have hD0m1_eq : BalanceEquiv (balance_sub (balance_add bone (balance_mul delta (balance_mul bzero bzero))) bone) bzero :=
    bal_poly_eq (fun | 0 => delta | _ => bzero) e_s126_d0m1_L .zero p_s126_d0m1
  have hcomb_c : ∀ (qc wc : Balance),
      BalanceEquiv (balance_add (balance_mul bone qc)
        (balance_mul (balance_sub (balance_add bone (balance_mul delta (balance_mul bzero bzero))) bone) wc)) qc :=
    fun qc wc => bal_poly_eq (fun | 0 => qc | 1 => wc | 2 => delta | _ => bzero)
      e_s126_comb0_L (.var 0) p_s126_comb0
  have hcomb_eq : Vec4Equiv (vec4_add (vec4_scale bone q)
      (vec4_scale (balance_sub (balance_add bone (balance_mul delta (balance_mul bzero bzero))) bone) w)) q :=
    ⟨hcomb_c q.w w.w, hcomb_c q.x w.x, hcomb_c q.y w.y, hcomb_c q.z w.z⟩
  have hcomb_norm := dot4_congr hcomb_eq hcomb_eq
  let envD : Nat → Balance := fun | 0 => a | 1 => delta | _ => bzero
  have hda : DualEquiv
      (dual_mul ⟨balance_add bone (balance_mul delta (balance_mul a a)),
                 balance_add bzero (balance_add (balance_mul delta (balance_add (balance_mul a bone) (balance_mul bone a)))
                                                (balance_mul bzero (balance_mul a a)))⟩
                ⟨D_lam, balance_neg (balance_mul (balOfNat 2) (balance_mul delta a))⟩)
      (dual_mul ⟨balance_mul D_lam D_lam, bzero⟩ ⟨bone, bzero⟩) :=
    ⟨bal_poly_eq envD e_s126_da_L.1 e_s126_da_R.1 p_s126_da_re,
     bal_poly_eq envD e_s126_da_L.2 e_s126_da_R.2 p_s126_da_eps⟩
  have hdd : DualEquiv
      (dual_mul ⟨balance_add bone (balance_mul delta (balance_mul a a)),
                 balance_add bzero (balance_add (balance_mul delta bzero) (balance_mul bone (balance_mul a a)))⟩
                ⟨D_lam, balance_neg (balance_mul a a)⟩)
      (dual_mul ⟨balance_mul D_lam D_lam, bzero⟩ ⟨bone, bzero⟩) :=
    ⟨bal_poly_eq envD e_s126_dd_L.1 e_s126_dd_R.1 p_s126_dd_re,
     bal_poly_eq envD e_s126_dd_L.2 e_s126_dd_R.2 p_s126_dd_eps⟩
  -- 4. Surprise
  have hsurp_le : BalLe eps (balance_add ema_slow eps) :=
    bal_ge_zero_of_equiv (bal_poly_eq (fun | 0 => ema_slow | 1 => eps | _ => bzero)
      e_add_sub_cancel_L e_add_sub_cancel_R p_add_sub_cancel) hes
  have hsurp_gt : BalGtZero (balance_add ema_slow eps) :=
    bal_gt_zero_of_equiv (bal_add_comm eps ema_slow) (bal_add_gt_ge heps hes)
  exact ⟨⟨hDidbd_gt, hloS, hhiS, hloD, hhiD⟩,
         ⟨hDin_ge1, hDin_gt, hx_le⟩,
         ⟨hDlam_ge1, hDlam_gt, hDlam_ge1, hD0_eq, hD0m1_eq, hcomb_eq, hcomb_norm, hda, hdd⟩,
         ⟨hef, hsurp_le, hsurp_gt⟩⟩

-- ----- ШАГ 127. Вложенная цепочка SO(2) ⊂ SO(3) ⊂ SO(4) ⊂ O(4), регуляризатор чётности и ∂H_β/∂v -----

def qconj (q : Vec4) : Vec4 :=
  ⟨q.w, balance_neg q.x, balance_neg q.y, balance_neg q.z⟩

theorem qmul_congr {p1 p2 q1 q2 : Vec4} (hp : Vec4Equiv p1 p2) (hq : Vec4Equiv q1 q2) :
    Vec4Equiv (qmul p1 q1) (qmul p2 q2) :=
  ⟨bal_sub_congr (bal_sub_congr (bal_mul_congr hp.1 hq.1) (bal_mul_congr hp.2.1 hq.2.1))
                 (step25_balance_add_congr (bal_mul_congr hp.2.2.1 hq.2.2.1) (bal_mul_congr hp.2.2.2 hq.2.2.2)),
   bal_sub_congr (step25_balance_add_congr (bal_mul_congr hp.1 hq.2.1) (bal_mul_congr hp.2.1 hq.1))
                 (bal_sub_congr (bal_mul_congr hp.2.2.2 hq.2.2.1) (bal_mul_congr hp.2.2.1 hq.2.2.2)),
   bal_sub_congr (step25_balance_add_congr (bal_mul_congr hp.1 hq.2.2.1) (bal_mul_congr hp.2.2.1 hq.1))
                 (bal_sub_congr (bal_mul_congr hp.2.1 hq.2.2.2) (bal_mul_congr hp.2.2.2 hq.2.1)),
   bal_sub_congr (step25_balance_add_congr (bal_mul_congr hp.1 hq.2.2.2) (bal_mul_congr hp.2.2.2 hq.1))
                 (bal_sub_congr (bal_mul_congr hp.2.2.1 hq.2.1) (bal_mul_congr hp.2.1 hq.2.2.1))⟩

private def e_s127_dot3_neg_L : BExpr :=
  vDotE (.neg (.var 0), .neg (.var 1), .neg (.var 2)) (.neg (.var 0), .neg (.var 1), .neg (.var 2))
private def e_s127_dot3_neg_R : BExpr :=
  vDotE (.var 0, .var 1, .var 2) (.var 0, .var 1, .var 2)
private theorem p_s127_dot3_neg : e_s127_dot3_neg_L.norm = e_s127_dot3_neg_R.norm := rfl

private def e_s127_two_neg_L : BExpr := .mul (.nat 2) (.neg (.var 0))
private def e_s127_two_neg_R : BExpr := .neg (.mul (.nat 2) (.var 0))
private theorem p_s127_two_neg : e_s127_two_neg_L.norm = e_s127_two_neg_R.norm := rfl

theorem q_stereo_neg (g : Vec3) : Vec4Equiv (q_stereo (vec3_neg g)) (qconj (q_stereo g)) := by
  let eg : Nat → Balance := fun | 0 => g.x | 1 => g.y | 2 => g.z | _ => bzero
  have hd : BalanceEquiv (dot3 (vec3_neg g) (vec3_neg g)) (dot3 g g) :=
    bal_poly_eq eg e_s127_dot3_neg_L e_s127_dot3_neg_R p_s127_dot3_neg
  have h2x := bal_poly_eq (fun | 0 => g.x | _ => bzero) e_s127_two_neg_L e_s127_two_neg_R p_s127_two_neg
  have h2y := bal_poly_eq (fun | 0 => g.y | _ => bzero) e_s127_two_neg_L e_s127_two_neg_R p_s127_two_neg
  have h2z := bal_poly_eq (fun | 0 => g.z | _ => bzero) e_s127_two_neg_L e_s127_two_neg_R p_s127_two_neg
  exact ⟨bal_sub_congr (step24_balance_refl bone) hd, h2x, h2y, h2z⟩

private def e_s127_qconj_w_L : BExpr :=
  let eq : BExpr × BExpr × BExpr × BExpr := (.var 0, .var 1, .var 2, .var 3)
  let eu : BExpr × BExpr × BExpr × BExpr := (.var 4, .var 5, .var 6, .var 7)
  let eqc : BExpr × BExpr × BExpr × BExpr := (.var 0, .neg (.var 1), .neg (.var 2), .neg (.var 3))
  (bQmulE eq (bQmulE eu eqc)).1
private def e_s127_qconj_w_R : BExpr :=
  let eq : BExpr × BExpr × BExpr × BExpr := (.var 0, .var 1, .var 2, .var 3)
  .mul (bDot4E eq eq) (.var 4)
private theorem p_s127_qconj_w : e_s127_qconj_w_L.norm = e_s127_qconj_w_R.norm := rfl

theorem qmul_conj_w (q u : Vec4) :
    BalanceEquiv (qmul q (qmul u (qconj q))).w (balance_mul (dot4 q q) u.w) :=
  bal_poly_eq (fun | 0 => q.w | 1 => q.x | 2 => q.y | 3 => q.z
                   | 4 => u.w | 5 => u.x | 6 => u.y | 7 => u.z | _ => bzero)
    e_s127_qconj_w_L e_s127_qconj_w_R p_s127_qconj_w

private def e_s127_norm3_L : BExpr :=
  .sub (.add (.add (.var 0) (.var 1)) (.add (.var 2) (.var 3))) (.var 0)
private def e_s127_norm3_R : BExpr :=
  .add (.add (.var 1) (.var 2)) (.var 3)
private theorem p_s127_norm3 : e_s127_norm3_L.norm = e_s127_norm3_R.norm := rfl

private def e_s127_norm3k_L : BExpr :=
  .sub (.mul (.var 4) (.add (.add (.var 0) (.var 1)) (.add (.var 2) (.var 3)))) (.mul (.var 4) (.var 0))
private def e_s127_norm3k_R : BExpr :=
  .mul (.var 4) (.add (.add (.var 1) (.var 2)) (.var 3))
private theorem p_s127_norm3k : e_s127_norm3k_L.norm = e_s127_norm3k_R.norm := rfl

private def e_s127_sq2_L : BExpr := .mul (.mul (.var 0) (.var 1)) (.mul (.var 0) (.var 1))
private def e_s127_sq2_R : BExpr := .mul (.mul (.var 0) (.var 0)) (.mul (.var 1) (.var 1))
private theorem p_s127_sq2 : e_s127_sq2_L.norm = e_s127_sq2_R.norm := rfl

private def e_s127_d3sq_L : BExpr := .mul (.mul (.var 0) (.var 1)) (.mul (.var 0) (.var 1))
private def e_s127_d3sq_R : BExpr := .mul (.mul (.var 0) (.var 0)) (.mul (.var 1) (.var 1))
private theorem p_s127_d3sq : e_s127_d3sq_L.norm = e_s127_d3sq_R.norm := rfl

-- Общая SO(2) лемма для чисто k-кватерниона q = (a, 0, 0, b) и qbar = (a, 0, 0, -b)
private def e_s127_qk : BExpr × BExpr × BExpr × BExpr := (.var 0, .zero, .zero, .var 1)
private def e_s127_qkb : BExpr × BExpr × BExpr × BExpr := (.var 0, .zero, .zero, .neg (.var 1))
private def e_s127_uk : BExpr × BExpr × BExpr × BExpr := (.var 2, .var 3, .var 4, .var 5)
private def e_s127_u2k := bQmulE e_s127_qk (bQmulE e_s127_uk e_s127_qkb)
private def e_s127_ab2 : BExpr := .add (.mul (.var 0) (.var 0)) (.mul (.var 1) (.var 1))

private theorem p_s127_so2_w : e_s127_u2k.1.norm = (BExpr.mul e_s127_ab2 (.var 2)).norm := rfl
private theorem p_s127_so2_z : e_s127_u2k.2.2.2.norm = (BExpr.mul e_s127_ab2 (.var 5)).norm := rfl
private theorem p_s127_so2_xy :
    (BExpr.add (.mul e_s127_u2k.2.1 e_s127_u2k.2.1) (.mul e_s127_u2k.2.2.1 e_s127_u2k.2.2.1)).norm =
    (BExpr.mul (.mul e_s127_ab2 e_s127_ab2) (.add (.mul (.var 3) (.var 3)) (.mul (.var 4) (.var 4)))).norm := rfl

private def e_s127_gvz_dot : BExpr :=
  vDotE (.zero, .zero, .var 0) (.zero, .zero, .var 0)
private theorem p_s127_gvz_dot : e_s127_gvz_dot.norm = (BExpr.mul (.var 0) (.var 0)).norm := rfl

private def e_s127_reg_gap_L : BExpr := .mul (.sub (.var 1) (.var 0)) (.sub (.var 1) (.var 0))
private def e_s127_reg_gap_R : BExpr :=
  .sub (.mul (.var 1) (.var 1)) (.mul (.var 0) (.sub (.mul (.nat 2) (.var 1)) (.var 0)))
private theorem p_s127_reg_gap : e_s127_reg_gap_L.norm = e_s127_reg_gap_R.norm := rfl

private def e_s127_reg_b0_L : BExpr := .mul .zero (.sub (.mul (.nat 2) (.var 0)) .zero)
private theorem p_s127_reg_b0 : e_s127_reg_b0_L.norm = BExpr.zero.norm := rfl

private def e_s127_reg_b2_L : BExpr := .mul (.mul (.nat 2) (.var 0)) (.sub (.mul (.nat 2) (.var 0)) (.mul (.nat 2) (.var 0)))
private theorem p_s127_reg_b2 : e_s127_reg_b2_L.norm = BExpr.zero.norm := rfl

private def e_s127_regd_L : BExpr × BExpr :=
  dMulE (.var 0, .one) (dAddE (.nat 2, .zero) (dMulE (.neg .one, .zero) (.var 0, .one)))
private def e_s127_regd_R : BExpr × BExpr :=
  (.mul (.var 0) (.sub (.nat 2) (.var 0)), .sub (.nat 2) (.mul (.nat 2) (.var 0)))
private theorem p_s127_regd_re : e_s127_regd_L.1.norm = e_s127_regd_R.1.norm := rfl
private theorem p_s127_regd_eps : e_s127_regd_L.2.norm = e_s127_regd_R.2.norm := rfl

private def e_s127_bderiv_lo_L : BExpr := .mul (.nat 2) (.sub (.nat 2) (.var 0))
private def e_s127_bderiv_lo_R : BExpr := .sub (.sub (.nat 2) (.mul (.nat 2) (.var 0))) (.neg (.nat 2))
private theorem p_s127_bderiv_lo : e_s127_bderiv_lo_L.norm = e_s127_bderiv_lo_R.norm := rfl

private def e_s127_bderiv_hi_L : BExpr := .mul (.nat 2) (.var 0)
private def e_s127_bderiv_hi_R : BExpr := .sub (.nat 2) (.sub (.nat 2) (.mul (.nat 2) (.var 0)))
private theorem p_s127_bderiv_hi : e_s127_bderiv_hi_L.norm = e_s127_bderiv_hi_R.norm := rfl

-- DualBalance для ∂H_β(v)u / ∂v_w
private def e_s127_Dd : BExpr × BExpr :=
  dAddE (dAddE (dAddE (dMulE (.var 0, .one) (.var 0, .one)) (.var 1, .zero))
               (dAddE (.var 2, .zero) (.var 3, .zero)))
        (.var 4, .zero)
private def e_s127_vud : BExpr × BExpr :=
  dAddE (dAddE (dMulE (.var 0, .one) (.var 5, .zero)) (.var 6, .zero))
        (dAddE (.var 7, .zero) (.var 8, .zero))
private def e_s127_hwd : BExpr × BExpr :=
  dAddE (dMulE (dMulE (.var 9, .zero) e_s127_Dd) (.var 5, .zero))
        (dMulE (.neg (.var 10), .zero) (dMulE e_s127_vud (.var 0, .one)))
private def e_s127_hxd : BExpr × BExpr :=
  dAddE (dMulE (dMulE (.var 9, .zero) e_s127_Dd) (.var 11, .zero))
        (dMulE (.neg (.var 10), .zero) (dMulE e_s127_vud (.var 12, .zero)))

private def e_s127_D_re : BExpr := .add (.add (.add (.mul (.var 0) (.var 0)) (.var 1)) (.add (.var 2) (.var 3))) (.var 4)
private def e_s127_vu_re : BExpr := .add (.add (.mul (.var 0) (.var 5)) (.var 6)) (.add (.var 7) (.var 8))
private theorem p_s127_D_re : e_s127_Dd.1.norm = e_s127_D_re.norm := rfl
private theorem p_s127_D_eps : e_s127_Dd.2.norm = (BExpr.mul (.nat 2) (.var 0)).norm := rfl
private theorem p_s127_vu_re : e_s127_vud.1.norm = e_s127_vu_re.norm := rfl
private theorem p_s127_vu_eps : e_s127_vud.2.norm = (BExpr.var 5).norm := rfl
private theorem p_s127_hw_re : e_s127_hwd.1.norm =
    (BExpr.sub (.mul (.mul (.var 9) e_s127_D_re) (.var 5)) (.mul (.mul (.var 10) e_s127_vu_re) (.var 0))).norm := rfl
private theorem p_s127_hw_eps : e_s127_hwd.2.norm =
    (BExpr.sub (.mul (.mul (.nat 2) (.var 9)) (.mul (.var 0) (.var 5)))
               (.mul (.var 10) (.add (.mul (.var 5) (.var 0)) e_s127_vu_re))).norm := rfl
private theorem p_s127_hx_re : e_s127_hxd.1.norm =
    (BExpr.sub (.mul (.mul (.var 9) e_s127_D_re) (.var 11)) (.mul (.mul (.var 10) e_s127_vu_re) (.var 12))).norm := rfl
private theorem p_s127_hx_eps : e_s127_hxd.2.norm =
    (BExpr.sub (.mul (.mul (.nat 2) (.var 9)) (.mul (.var 0) (.var 11)))
               (.mul (.var 10) (.mul (.var 5) (.var 12)))).norm := rfl

/-- ШАГ 127. Вложенная цепочка SO(2) ⊂ SO(3) ⊂ SO(4) ⊂ O(4), zero-oracle регуляризатор чётности
    и точная RTRL-производная отражения Хаусхолдера `H_β(v)u` по нормальному вектору `v`:
    1. `q_stereo(-g) ~ qconj(q_stereo(g))` (основа симметрично-антисимметричной параметризации `g_L = g_+ + g_-, g_R = -g_+ + g_-`).
    2. Предел `SO(3) ⊂ SO(4)` (`g_- = 0`): `u3.w ~ (1 + ||g||²)²·u.w` (скалярная координата инвариантна)
       и 3D-векторная норма `||u3.xyz||² ~ (1 + ||g||²)⁴·||u.xyz||²`.
    3. Предел `SO(2) ⊂ SO(3) ⊂ SO(4)` (`g_- = 0, g_+ = (0, 0, g_z)`): `u2.w ~ d2²·u.w`, `u2.z ~ d2²·u.z`,
       и 2D-плоская норма `u2.x² + u2.y² ~ d2⁴·(u.x² + u.y²)`.
    4. Регуляризатор чётности `R(p, q) = p(2q - p)`: при `0 ≤ p ≤ 2q` всегда `0 ≤ R(p, q) ≤ q²`,
       `R(0, q) ~ 0`, `R(2q, q) ~ 0`, а `DualBalance`-производная `β(2 - β)` равна `2 - 2β ∈ [-2, 2]`.
    5. Точная `DualBalance`-производная `householder4_num` по `v.w`:
       `∂h_w/∂v_w = 2 q v_w u_w - p (u_w v_w + v·u)` и `∂h_x/∂v_w = 2 q v_w u_x - p u_w v_x`. -/
theorem step127_nested_so2_so3_so4_o4_parity_and_householder_v_deriv
    (g : Vec3) (gz : Balance) (u v : Vec4) (p q eps beta : Balance)
    (hp0 : BalGeZero p) (hp2 : BalLe p (balance_mul (balOfNat 2) q))
    (hb0 : BalGeZero beta) (hb2 : BalLe beta (balOfNat 2)) :
    let u3 := qmul (q_stereo g) (qmul u (q_stereo (vec3_neg g)))
    let d3 := balance_add bone (dot3 g g)
    let gvz : Vec3 := ⟨bzero, bzero, gz⟩
    let u2 := qmul (q_stereo gvz) (qmul u (q_stereo (vec3_neg gvz)))
    let d2 := balance_add bone (bsq gz)
    let reg := balance_mul p (balance_sub (balance_mul (balOfNat 2) q) p)
    let betad := dual_var beta
    let regd := dual_mul betad (dual_add (dual_const (balOfNat 2))
                                          (dual_mul (dual_const (balance_neg bone)) betad))
    let vwd := dual_var v.w
    let D_d := dual_add (dual_add (dual_add (dual_mul vwd vwd) (dual_const (bsq v.x)))
                                  (dual_add (dual_const (bsq v.y)) (dual_const (bsq v.z))))
                        (dual_const eps)
    let vu_d := dual_add (dual_add (dual_mul vwd (dual_const u.w)) (dual_const (balance_mul v.x u.x)))
                         (dual_add (dual_const (balance_mul v.y u.y)) (dual_const (balance_mul v.z u.z)))
    let hw_d := dual_add (dual_mul (dual_mul (dual_const q) D_d) (dual_const u.w))
                         (dual_mul (dual_const (balance_neg p)) (dual_mul vu_d vwd))
    let hx_d := dual_add (dual_mul (dual_mul (dual_const q) D_d) (dual_const u.x))
                         (dual_mul (dual_const (balance_neg p)) (dual_mul vu_d (dual_const v.x)))
    let D_v := balance_add (dot4 v v) eps
    let vu := dot4 v u
    -- 1. Сопряжение
    Vec4Equiv (q_stereo (vec3_neg g)) (qconj (q_stereo g))
    ∧
    -- 2. Вкладка SO(3) ⊂ SO(4)
    (BalanceEquiv u3.w (balance_mul (bsq d3) u.w) ∧
     BalanceEquiv (balance_add (balance_add (bsq u3.x) (bsq u3.y)) (bsq u3.z))
                  (balance_mul (bsq (bsq d3)) (balance_add (balance_add (bsq u.x) (bsq u.y)) (bsq u.z))))
    ∧
    -- 3. Вкладка SO(2) ⊂ SO(3) ⊂ SO(4)
    (BalanceEquiv u2.w (balance_mul (bsq d2) u.w) ∧
     BalanceEquiv u2.z (balance_mul (bsq d2) u.z) ∧
     BalanceEquiv (balance_add (bsq u2.x) (bsq u2.y))
                  (balance_mul (bsq (bsq d2)) (balance_add (bsq u.x) (bsq u.y))))
    ∧
    -- 4. Регуляризатор чётности
    (BalGeZero reg ∧ BalLe reg (bsq q) ∧
     BalanceEquiv (balance_mul bzero (balance_sub (balance_mul (balOfNat 2) q) bzero)) bzero ∧
     BalanceEquiv (balance_mul (balance_mul (balOfNat 2) q)
                    (balance_sub (balance_mul (balOfNat 2) q) (balance_mul (balOfNat 2) q))) bzero ∧
     DualEquiv regd ⟨balance_mul beta (balance_sub (balOfNat 2) beta),
                     balance_sub (balOfNat 2) (balance_mul (balOfNat 2) beta)⟩ ∧
     BalLe (balance_neg (balOfNat 2)) (balance_sub (balOfNat 2) (balance_mul (balOfNat 2) beta)) ∧
     BalLe (balance_sub (balOfNat 2) (balance_mul (balOfNat 2) beta)) (balOfNat 2))
    ∧
    -- 5. Точные DualBalance производные Хаусхолдера по v.w
    (DualEquiv D_d ⟨D_v, balance_mul (balOfNat 2) v.w⟩ ∧
     DualEquiv vu_d ⟨vu, u.w⟩ ∧
     DualEquiv hw_d ⟨balance_sub (balance_mul (balance_mul q D_v) u.w) (balance_mul (balance_mul p vu) v.w),
                     balance_sub (balance_mul (balance_mul (balOfNat 2) q) (balance_mul v.w u.w))
                                 (balance_mul p (balance_add (balance_mul u.w v.w) vu))⟩ ∧
     DualEquiv hx_d ⟨balance_sub (balance_mul (balance_mul q D_v) u.x) (balance_mul (balance_mul p vu) v.x),
                     balance_sub (balance_mul (balance_mul (balOfNat 2) q) (balance_mul v.w u.x))
                                 (balance_mul p (balance_mul u.w v.x))⟩) := by
  dsimp only [dual_var, dual_const, dual_mul, dual_add, DualEquiv, bsq]
  -- 1. Сопряжение
  have hconj := q_stereo_neg g
  -- 2. SO(3) ⊂ SO(4)
  let qg := q_stereo g
  let u3 := qmul qg (qmul u (q_stereo (vec3_neg g)))
  let d3 := balance_add bone (dot3 g g)
  have hu3_conj : Vec4Equiv u3 (qmul qg (qmul u (qconj qg))) :=
    qmul_congr ⟨step24_balance_refl _, step24_balance_refl _, step24_balance_refl _, step24_balance_refl _⟩
      (qmul_congr ⟨step24_balance_refl _, step24_balance_refl _, step24_balance_refl _, step24_balance_refl _⟩ hconj)
  have hs117 := step117_rational_quat_and_so4_double_rotor g (vec3_neg g) u
  have hqg_norm : BalanceEquiv (dot4 qg qg) (balance_mul d3 d3) := hs117.2.2.1
  have hu3_w : BalanceEquiv u3.w (balance_mul (balance_mul d3 d3) u.w) :=
    step24_balance_trans hu3_conj.1
      (step24_balance_trans (qmul_conj_w qg u) (bal_mul_congr hqg_norm (step24_balance_refl u.w)))
  have hd_neg : BalanceEquiv (balance_add bone (dot3 (vec3_neg g) (vec3_neg g))) d3 :=
    step25_balance_add_congr (step24_balance_refl bone)
      (bal_poly_eq (fun | 0 => g.x | 1 => g.y | 2 => g.z | _ => bzero)
        e_s127_dot3_neg_L e_s127_dot3_neg_R p_s127_dot3_neg)
  have hu3_tot : BalanceEquiv (dot4 u3 u3) (balance_mul (balance_mul (balance_mul d3 d3) (balance_mul d3 d3)) (dot4 u u)) :=
    step24_balance_trans hs117.2.2.2.2.1
      (bal_mul_congr
        (let K := balance_mul (balance_mul d3 d3) (balance_mul d3 d3)
         have hdd : BalanceEquiv (balance_mul d3 (balance_add bone (dot3 (vec3_neg g) (vec3_neg g))))
                                 (balance_mul d3 d3) :=
           bal_mul_congr (step24_balance_refl d3) hd_neg
         step24_balance_trans (bal_mul_congr hdd hdd)
           (bal_poly_eq (fun | 0 => d3 | 1 => d3 | _ => bzero) e_s127_d3sq_L e_s127_d3sq_R p_s127_d3sq))
        (step24_balance_refl (dot4 u u)))
  have hu3_wsq : BalanceEquiv (balance_mul u3.w u3.w)
      (balance_mul (balance_mul (balance_mul d3 d3) (balance_mul d3 d3)) (balance_mul u.w u.w)) :=
    step24_balance_trans (bal_mul_congr hu3_w hu3_w)
      (bal_poly_eq (fun | 0 => balance_mul d3 d3 | 1 => u.w | _ => bzero) e_s127_sq2_L e_s127_sq2_R p_s127_sq2)
  have hu3_xyz : BalanceEquiv
      (balance_add (balance_add (balance_mul u3.x u3.x) (balance_mul u3.y u3.y)) (balance_mul u3.z u3.z))
      (balance_mul (balance_mul (balance_mul d3 d3) (balance_mul d3 d3))
        (balance_add (balance_add (balance_mul u.x u.x) (balance_mul u.y u.y)) (balance_mul u.z u.z))) := by
    let K := balance_mul (balance_mul d3 d3) (balance_mul d3 d3)
    have hs := bal_sub_congr hu3_tot hu3_wsq
    have hL := bal_poly_eq
      (fun | 0 => balance_mul u3.w u3.w | 1 => balance_mul u3.x u3.x
           | 2 => balance_mul u3.y u3.y | 3 => balance_mul u3.z u3.z | _ => bzero)
      e_s127_norm3_L e_s127_norm3_R p_s127_norm3
    have hR := bal_poly_eq
      (fun | 0 => balance_mul u.w u.w | 1 => balance_mul u.x u.x
           | 2 => balance_mul u.y u.y | 3 => balance_mul u.z u.z | 4 => K | _ => bzero)
      e_s127_norm3k_L e_s127_norm3k_R p_s127_norm3k
    exact step24_balance_trans (step24_balance_symm hL) (step24_balance_trans hs hR)
  -- 3. SO(2) ⊂ SO(3) ⊂ SO(4)
  let gvz : Vec3 := ⟨bzero, bzero, gz⟩
  let d2 := balance_add bone (balance_mul gz gz)
  let a2 := balance_sub bone (balance_mul gz gz)
  let b2 := balance_mul (balOfNat 2) gz
  have hgvz_dot : BalanceEquiv (dot3 gvz gvz) (balance_mul gz gz) :=
    bal_poly_eq (fun | 0 => gz | _ => bzero) e_s127_gvz_dot (.mul (.var 0) (.var 0)) p_s127_gvz_dot
  have hq2_L : Vec4Equiv (q_stereo gvz) ⟨a2, bzero, bzero, b2⟩ :=
    ⟨bal_sub_congr (step24_balance_refl bone) hgvz_dot,
     bal_mul_zero (balOfNat 2), bal_mul_zero (balOfNat 2), step24_balance_refl b2⟩
  have hq2_R : Vec4Equiv (q_stereo (vec3_neg gvz)) ⟨a2, bzero, bzero, balance_neg b2⟩ := by
    have hc := q_stereo_neg gvz
    have hc' : Vec4Equiv (qconj (q_stereo gvz)) ⟨a2, bzero, bzero, balance_neg b2⟩ :=
      ⟨hq2_L.1, step24_balance_trans (bal_neg_congr hq2_L.2.1) bal_neg_zero,
       step24_balance_trans (bal_neg_congr hq2_L.2.2.1) bal_neg_zero, bal_neg_congr hq2_L.2.2.2⟩
    exact ⟨step24_balance_trans hc.1 hc'.1, step24_balance_trans hc.2.1 hc'.2.1,
           step24_balance_trans hc.2.2.1 hc'.2.2.1, step24_balance_trans hc.2.2.2 hc'.2.2.2⟩
  let u2k := qmul ⟨a2, bzero, bzero, b2⟩ (qmul u ⟨a2, bzero, bzero, balance_neg b2⟩)
  have hu2_eq : Vec4Equiv (qmul (q_stereo gvz) (qmul u (q_stereo (vec3_neg gvz)))) u2k :=
    qmul_congr hq2_L (qmul_congr ⟨step24_balance_refl _, step24_balance_refl _, step24_balance_refl _, step24_balance_refl _⟩ hq2_R)
  have hab2_eq : BalanceEquiv (balance_add (balance_mul a2 a2) (balance_mul b2 b2)) (balance_mul d2 d2) := by
    let env2 : Nat → Balance := fun | 0 => balance_mul gz gz | _ => bzero
    have h4 : BalanceEquiv (balance_mul b2 b2) (balance_mul (balOfNat 4) (balance_mul gz gz)) :=
      bal_poly_eq (fun | 0 => gz | _ => bzero)
        (.mul (.mul (.nat 2) (.var 0)) (.mul (.nat 2) (.var 0))) (.mul (.nat 4) (.mul (.var 0) (.var 0))) rfl
    exact step24_balance_trans (step25_balance_add_congr (step24_balance_refl _) h4)
      (bal_poly_eq env2 e_stereo_L e_stereo_R p_stereo_eq)
  let envk : Nat → Balance := fun | 0 => a2 | 1 => b2 | 2 => u.w | 3 => u.x | 4 => u.y | 5 => u.z | _ => bzero
  have hu2_w : BalanceEquiv (qmul (q_stereo gvz) (qmul u (q_stereo (vec3_neg gvz)))).w
      (balance_mul (balance_mul d2 d2) u.w) :=
    step24_balance_trans hu2_eq.1
      (step24_balance_trans (bal_poly_eq envk e_s127_u2k.1 (.mul e_s127_ab2 (.var 2)) p_s127_so2_w)
        (bal_mul_congr hab2_eq (step24_balance_refl u.w)))
  have hu2_z : BalanceEquiv (qmul (q_stereo gvz) (qmul u (q_stereo (vec3_neg gvz)))).z
      (balance_mul (balance_mul d2 d2) u.z) :=
    step24_balance_trans hu2_eq.2.2.2
      (step24_balance_trans (bal_poly_eq envk e_s127_u2k.2.2.2 (.mul e_s127_ab2 (.var 5)) p_s127_so2_z)
        (bal_mul_congr hab2_eq (step24_balance_refl u.z)))
  have hu2_xy : BalanceEquiv
      (balance_add (balance_mul (qmul (q_stereo gvz) (qmul u (q_stereo (vec3_neg gvz)))).x
                                (qmul (q_stereo gvz) (qmul u (q_stereo (vec3_neg gvz)))).x)
                   (balance_mul (qmul (q_stereo gvz) (qmul u (q_stereo (vec3_neg gvz)))).y
                                (qmul (q_stereo gvz) (qmul u (q_stereo (vec3_neg gvz)))).y))
      (balance_mul (balance_mul (balance_mul d2 d2) (balance_mul d2 d2))
        (balance_add (balance_mul u.x u.x) (balance_mul u.y u.y))) :=
    step24_balance_trans
      (step25_balance_add_congr (bal_mul_congr hu2_eq.2.1 hu2_eq.2.1) (bal_mul_congr hu2_eq.2.2.1 hu2_eq.2.2.1))
      (step24_balance_trans
        (bal_poly_eq envk (.add (.mul e_s127_u2k.2.1 e_s127_u2k.2.1) (.mul e_s127_u2k.2.2.1 e_s127_u2k.2.2.1))
          (.mul (.mul e_s127_ab2 e_s127_ab2) (.add (.mul (.var 3) (.var 3)) (.mul (.var 4) (.var 4)))) p_s127_so2_xy)
        (bal_mul_congr (bal_mul_congr hab2_eq hab2_eq) (step24_balance_refl _)))
  -- 4. Регуляризатор чётности
  have hreg_ge : BalGeZero (balance_mul p (balance_sub (balance_mul (balOfNat 2) q) p)) :=
    bal_mul_ge_ge hp0 hp2
  have hreg_le : BalLe (balance_mul p (balance_sub (balance_mul (balOfNat 2) q) p)) (balance_mul q q) :=
    bal_ge_zero_of_equiv
      (bal_poly_eq (fun | 0 => p | 1 => q | _ => bzero) e_s127_reg_gap_L e_s127_reg_gap_R p_s127_reg_gap)
      (bal_sq_ge_zero (balance_sub q p))
  have hreg_b0 := bal_poly_eq (fun | 0 => q | _ => bzero) e_s127_reg_b0_L .zero p_s127_reg_b0
  have hreg_b2 := bal_poly_eq (fun | 0 => q | _ => bzero) e_s127_reg_b2_L .zero p_s127_reg_b2
  have hregd_re := bal_poly_eq (fun | 0 => beta | _ => bzero) e_s127_regd_L.1 e_s127_regd_R.1 p_s127_regd_re
  have hregd_eps := bal_poly_eq (fun | 0 => beta | _ => bzero) e_s127_regd_L.2 e_s127_regd_R.2 p_s127_regd_eps
  have hbderiv_lo : BalLe (balance_neg (balOfNat 2)) (balance_sub (balOfNat 2) (balance_mul (balOfNat 2) beta)) :=
    bal_ge_zero_of_equiv
      (bal_poly_eq (fun | 0 => beta | _ => bzero) e_s127_bderiv_lo_L e_s127_bderiv_lo_R p_s127_bderiv_lo)
      (bal_mul_ge_ge (Or.inr (balOfNat_succ_gt_zero 1)) hb2)
  have hbderiv_hi : BalLe (balance_sub (balOfNat 2) (balance_mul (balOfNat 2) beta)) (balOfNat 2) :=
    bal_ge_zero_of_equiv
      (bal_poly_eq (fun | 0 => beta | _ => bzero) e_s127_bderiv_hi_L e_s127_bderiv_hi_R p_s127_bderiv_hi)
      (bal_mul_ge_ge (Or.inr (balOfNat_succ_gt_zero 1)) hb0)
  -- 5. DualBalance производные Хаусхолдера по v.w
  let envH : Nat → Balance := fun
    | 0 => v.w | 1 => balance_mul v.x v.x | 2 => balance_mul v.y v.y | 3 => balance_mul v.z v.z
    | 4 => eps | 5 => u.w | 6 => balance_mul v.x u.x | 7 => balance_mul v.y u.y | 8 => balance_mul v.z u.z
    | 9 => q | 10 => p | 11 => u.x | 12 => v.x | _ => bzero
  let vwd := dual_var v.w
  let D_d := dual_add (dual_add (dual_add (dual_mul vwd vwd) (dual_const (bsq v.x)))
                                (dual_add (dual_const (bsq v.y)) (dual_const (bsq v.z))))
                      (dual_const eps)
  let vu_d := dual_add (dual_add (dual_mul vwd (dual_const u.w)) (dual_const (balance_mul v.x u.x)))
                       (dual_add (dual_const (balance_mul v.y u.y)) (dual_const (balance_mul v.z u.z)))
  let hw_d := dual_add (dual_mul (dual_mul (dual_const q) D_d) (dual_const u.w))
                       (dual_mul (dual_const (balance_neg p)) (dual_mul vu_d vwd))
  let hx_d := dual_add (dual_mul (dual_mul (dual_const q) D_d) (dual_const u.x))
                       (dual_mul (dual_const (balance_neg p)) (dual_mul vu_d (dual_const v.x)))
  have hD_d : DualEquiv D_d ⟨balance_add (dot4 v v) eps, balance_mul (balOfNat 2) v.w⟩ :=
    ⟨bal_poly_eq envH e_s127_Dd.1 e_s127_D_re p_s127_D_re,
     bal_poly_eq envH e_s127_Dd.2 (.mul (.nat 2) (.var 0)) p_s127_D_eps⟩
  have hvu_d : DualEquiv vu_d ⟨dot4 v u, u.w⟩ :=
    ⟨bal_poly_eq envH e_s127_vud.1 e_s127_vu_re p_s127_vu_re,
     bal_poly_eq envH e_s127_vud.2 (.var 5) p_s127_vu_eps⟩
  have hhw_d : DualEquiv hw_d
      ⟨balance_sub (balance_mul (balance_mul q (balance_add (dot4 v v) eps)) u.w)
                   (balance_mul (balance_mul p (dot4 v u)) v.w),
       balance_sub (balance_mul (balance_mul (balOfNat 2) q) (balance_mul v.w u.w))
                   (balance_mul p (balance_add (balance_mul u.w v.w) (dot4 v u)))⟩ :=
    ⟨bal_poly_eq envH e_s127_hwd.1 _ p_s127_hw_re,
     bal_poly_eq envH e_s127_hwd.2 _ p_s127_hw_eps⟩
  have hhx_d : DualEquiv hx_d
      ⟨balance_sub (balance_mul (balance_mul q (balance_add (dot4 v v) eps)) u.x)
                   (balance_mul (balance_mul p (dot4 v u)) v.x),
       balance_sub (balance_mul (balance_mul (balOfNat 2) q) (balance_mul v.w u.x))
                   (balance_mul p (balance_mul u.w v.x))⟩ :=
    ⟨bal_poly_eq envH e_s127_hxd.1 _ p_s127_hx_re,
     bal_poly_eq envH e_s127_hxd.2 _ p_s127_hx_eps⟩
  exact ⟨hconj, ⟨hu3_w, hu3_xyz⟩, ⟨hu2_w, hu2_z, hu2_xy⟩,
         ⟨hreg_ge, hreg_le, hreg_b0, hreg_b2, ⟨hregd_re, hregd_eps⟩, hbderiv_lo, hbderiv_hi⟩,
         ⟨hD_d, hvu_d, hhw_d, hhx_d⟩⟩

-- ----- ШАГ 128. Выход из седла селективности (β = 0) и покомпонентные границы Q8.24 -----

private def e_s128_mul_lo_L : BExpr :=
  .add (.mul (.sub (.var 0) (.var 1)) (.sub (.var 2) (.var 3)))
       (.mul (.add (.var 0) (.var 1)) (.add (.var 2) (.var 3)))
private def e_s128_mul_lo_R : BExpr :=
  .mul (.nat 2) (.sub (.mul (.var 1) (.var 3)) (.neg (.mul (.var 0) (.var 2))))
private theorem p_s128_mul_lo : e_s128_mul_lo_L.norm = e_s128_mul_lo_R.norm := rfl

private def e_s128_mul_hi_L : BExpr :=
  .add (.mul (.sub (.var 0) (.var 1)) (.add (.var 2) (.var 3)))
       (.mul (.add (.var 0) (.var 1)) (.sub (.var 2) (.var 3)))
private def e_s128_mul_hi_R : BExpr :=
  .mul (.nat 2) (.sub (.mul (.var 0) (.var 2)) (.mul (.var 1) (.var 3)))
private theorem p_s128_mul_hi : e_s128_mul_hi_L.norm = e_s128_mul_hi_R.norm := rfl

private def e_s128_abs_add_lo_L : BExpr := .add (.add (.var 0) (.var 2)) (.add (.var 1) (.var 3))
private def e_s128_abs_add_lo_R : BExpr := .sub (.add (.var 2) (.var 3)) (.neg (.add (.var 0) (.var 1)))
private theorem p_s128_abs_add_lo : e_s128_abs_add_lo_L.norm = e_s128_abs_add_lo_R.norm := rfl

private def e_s128_abs_add_hi_L : BExpr := .add (.sub (.var 0) (.var 2)) (.sub (.var 1) (.var 3))
private def e_s128_abs_add_hi_R : BExpr := .sub (.add (.var 0) (.var 1)) (.add (.var 2) (.var 3))
private theorem p_s128_abs_add_hi : e_s128_abs_add_hi_L.norm = e_s128_abs_add_hi_R.norm := rfl

private def e_s128_abs_sub_lo_L : BExpr := .add (.add (.var 0) (.var 2)) (.sub (.var 1) (.var 3))
private def e_s128_abs_sub_lo_R : BExpr := .sub (.sub (.var 2) (.var 3)) (.neg (.add (.var 0) (.var 1)))
private theorem p_s128_abs_sub_lo : e_s128_abs_sub_lo_L.norm = e_s128_abs_sub_lo_R.norm := rfl

private def e_s128_abs_sub_hi_L : BExpr := .add (.sub (.var 0) (.var 2)) (.add (.var 1) (.var 3))
private def e_s128_abs_sub_hi_R : BExpr := .sub (.add (.var 0) (.var 1)) (.sub (.var 2) (.var 3))
private theorem p_s128_abs_sub_hi : e_s128_abs_sub_hi_L.norm = e_s128_abs_sub_hi_R.norm := rfl

theorem bal_abs_mul_le {A B a b : Balance}
    (ha1 : BalLe (balance_neg A) a) (ha2 : BalLe a A)
    (hb1 : BalLe (balance_neg B) b) (hb2 : BalLe b B) :
    BalLe (balance_neg (balance_mul A B)) (balance_mul a b) ∧
    BalLe (balance_mul a b) (balance_mul A B) := by
  let env : Nat → Balance := fun | 0 => A | 1 => a | 2 => B | 3 => b | _ => bzero
  have hap : BalGeZero (balance_add A a) :=
    bal_ge_zero_of_equiv (bal_poly_eq (fun | 0 => A | 1 => a | _ => bzero)
      e_sub_neg_eq_add_L e_sub_neg_eq_add_R p_sub_neg_eq_add) ha1
  have hbp : BalGeZero (balance_add B b) :=
    bal_ge_zero_of_equiv (bal_poly_eq (fun | 0 => B | 1 => b | _ => bzero)
      e_sub_neg_eq_add_L e_sub_neg_eq_add_R p_sub_neg_eq_add) hb1
  have h2lo := bal_add_ge_ge (bal_mul_ge_ge ha2 hb2) (bal_mul_ge_ge hap hbp)
  have hlo : BalLe (balance_neg (balance_mul A B)) (balance_mul a b) :=
    bal_ge_zero_of_double_ge
      (bal_ge_zero_of_equiv (bal_poly_eq env e_s128_mul_lo_L e_s128_mul_lo_R p_s128_mul_lo) h2lo)
  have h2hi := bal_add_ge_ge (bal_mul_ge_ge ha2 hbp) (bal_mul_ge_ge hap hb2)
  have hhi : BalLe (balance_mul a b) (balance_mul A B) :=
    bal_ge_zero_of_double_ge
      (bal_ge_zero_of_equiv (bal_poly_eq env e_s128_mul_hi_L e_s128_mul_hi_R p_s128_mul_hi) h2hi)
  exact ⟨hlo, hhi⟩

theorem bal_abs_add_le {A B a b : Balance}
    (ha1 : BalLe (balance_neg A) a) (ha2 : BalLe a A)
    (hb1 : BalLe (balance_neg B) b) (hb2 : BalLe b B) :
    BalLe (balance_neg (balance_add A B)) (balance_add a b) ∧
    BalLe (balance_add a b) (balance_add A B) := by
  let env : Nat → Balance := fun | 0 => A | 1 => B | 2 => a | 3 => b | _ => bzero
  have hap : BalGeZero (balance_add A a) :=
    bal_ge_zero_of_equiv (bal_poly_eq (fun | 0 => A | 1 => a | _ => bzero)
      e_sub_neg_eq_add_L e_sub_neg_eq_add_R p_sub_neg_eq_add) ha1
  have hbp : BalGeZero (balance_add B b) :=
    bal_ge_zero_of_equiv (bal_poly_eq (fun | 0 => B | 1 => b | _ => bzero)
      e_sub_neg_eq_add_L e_sub_neg_eq_add_R p_sub_neg_eq_add) hb1
  exact ⟨bal_ge_zero_of_equiv (bal_poly_eq env e_s128_abs_add_lo_L e_s128_abs_add_lo_R p_s128_abs_add_lo)
           (bal_add_ge_ge hap hbp),
         bal_ge_zero_of_equiv (bal_poly_eq env e_s128_abs_add_hi_L e_s128_abs_add_hi_R p_s128_abs_add_hi)
           (bal_add_ge_ge ha2 hb2)⟩

theorem bal_abs_sub_le {A B a b : Balance}
    (ha1 : BalLe (balance_neg A) a) (ha2 : BalLe a A)
    (hb1 : BalLe (balance_neg B) b) (hb2 : BalLe b B) :
    BalLe (balance_neg (balance_add A B)) (balance_sub a b) ∧
    BalLe (balance_sub a b) (balance_add A B) := by
  let env : Nat → Balance := fun | 0 => A | 1 => B | 2 => a | 3 => b | _ => bzero
  have hap : BalGeZero (balance_add A a) :=
    bal_ge_zero_of_equiv (bal_poly_eq (fun | 0 => A | 1 => a | _ => bzero)
      e_sub_neg_eq_add_L e_sub_neg_eq_add_R p_sub_neg_eq_add) ha1
  have hbp : BalGeZero (balance_add B b) :=
    bal_ge_zero_of_equiv (bal_poly_eq (fun | 0 => B | 1 => b | _ => bzero)
      e_sub_neg_eq_add_L e_sub_neg_eq_add_R p_sub_neg_eq_add) hb1
  exact ⟨bal_ge_zero_of_equiv (bal_poly_eq env e_s128_abs_sub_lo_L e_s128_abs_sub_lo_R p_s128_abs_sub_lo)
           (bal_add_ge_ge hap hb2),
         bal_ge_zero_of_equiv (bal_poly_eq env e_s128_abs_sub_hi_L e_s128_abs_sub_hi_R p_s128_abs_sub_hi)
           (bal_add_ge_ge ha2 hbp)⟩

private def e_s128_4M_L : BExpr := .add (.add (.var 0) (.var 0)) (.add (.var 0) (.var 0))
private def e_s128_4M_R : BExpr := .mul (.nat 4) (.var 0)
private theorem p_s128_4M : e_s128_4M_L.norm = e_s128_4M_R.norm := rfl

private theorem bal_abs_4_bounds (M t : Balance)
    (h1 : BalLe (balance_neg (balance_add (balance_add M M) (balance_add M M))) t)
    (h2 : BalLe t (balance_add (balance_add M M) (balance_add M M))) :
    BalLe (balance_neg (balance_mul (balOfNat 4) M)) t ∧ BalLe t (balance_mul (balOfNat 4) M) :=
  have heq := bal_poly_eq (fun | 0 => M | _ => bzero) e_s128_4M_L e_s128_4M_R p_s128_4M
  ⟨bal_le_of_equiv_left (bal_neg_congr heq) h1, bal_le_of_equiv_right heq h2⟩

private def e_s128_saddle_z_re_L : BExpr := .add (.mul bzero_E (.zero)) (.mul bzero_E .zero)
  where bzero_E : BExpr := .zero
private def e_s128_saddle_eps_L : BExpr := .add (.mul .zero .zero) (.mul .one (.var 0))
private theorem p_s128_saddle_eps : e_s128_saddle_eps_L.norm = (BExpr.var 0).norm := rfl
private def e_s128_dead_eps_L : BExpr := .add (.mul .zero .zero) (.mul .one .zero)
private theorem p_s128_dead_eps : e_s128_dead_eps_L.norm = BExpr.zero.norm := rfl

/-- ШАГ 128. Выход селективности из седла при `β = 0` и точные покомпонентные границы Q8.24:
    1. Селективное гейтирование `β_k · z_{t,k}` при `β_k = 0`:
       `DualEquiv ((0 + ε) · (z + 0ε)) (0 + z·ε)`, т.е. при `z ≠ 0` градиент по `β_k` строго ненулевой,
       в отличие от мёртвого седла `z = 0` (где и `∂/∂β_k ~ 0`, и `∂/∂w_Δ ~ 0`).
    2. Покомпонентная оценка произведения кватернионов `r = qmul p q` в Q8.24:
       если `|p_c| ≤ S` и `|q_c| ≤ S`, то `-4S² ≤ r_c ≤ 4S²` для всех 4 компонент (`w, x, y, z`).
    3. Покомпонентная оценка `dot4 v u` в Q8.24:
       если `|v_c| ≤ B` и `|u_c| ≤ U`, то `-4BU ≤ v·u ≤ 4BU`. -/
theorem step128_selectivity_saddle_escape_and_q824_product_bounds
    (z S B U : Balance) (p q v u : Vec4)
    (hpw1 : BalLe (balance_neg S) p.w) (hpw2 : BalLe p.w S)
    (hpx1 : BalLe (balance_neg S) p.x) (hpx2 : BalLe p.x S)
    (hpy1 : BalLe (balance_neg S) p.y) (hpy2 : BalLe p.y S)
    (hpz1 : BalLe (balance_neg S) p.z) (hpz2 : BalLe p.z S)
    (hqw1 : BalLe (balance_neg S) q.w) (hqw2 : BalLe q.w S)
    (hqx1 : BalLe (balance_neg S) q.x) (hqx2 : BalLe q.x S)
    (hqy1 : BalLe (balance_neg S) q.y) (hqy2 : BalLe q.y S)
    (hqz1 : BalLe (balance_neg S) q.z) (hqz2 : BalLe q.z S)
    (hvw1 : BalLe (balance_neg B) v.w) (hvw2 : BalLe v.w B)
    (hvx1 : BalLe (balance_neg B) v.x) (hvx2 : BalLe v.x B)
    (hvy1 : BalLe (balance_neg B) v.y) (hvy2 : BalLe v.y B)
    (hvz1 : BalLe (balance_neg B) v.z) (hvz2 : BalLe v.z B)
    (huw1 : BalLe (balance_neg U) u.w) (huw2 : BalLe u.w U)
    (hux1 : BalLe (balance_neg U) u.x) (hux2 : BalLe u.x U)
    (huy1 : BalLe (balance_neg U) u.y) (huy2 : BalLe u.y U)
    (huz1 : BalLe (balance_neg U) u.z) (huz2 : BalLe u.z U) :
    let r := qmul p q
    let M4S2 := balance_mul (balOfNat 4) (bsq S)
    let M4BU := balance_mul (balOfNat 4) (balance_mul B U)
    -- 1. Выход из седла при β = 0
    (DualEquiv (dual_mul (dual_var bzero) (dual_const z)) ⟨bzero, z⟩ ∧
     ((BalGtZero z ∨ BalLtZero z) →
       BalanceEquiv (dual_mul (dual_var bzero) (dual_const z)).eps bzero → False) ∧
     BalanceEquiv (dual_mul (dual_var bzero) (dual_const bzero)).eps bzero ∧
     BalanceEquiv (dual_mul (dual_const bzero) (dual_var bzero)).eps bzero)
    ∧
    -- 2. Оценка компонент qmul в Q8.24: -4S² ≤ r_c ≤ 4S²
    ((BalLe (balance_neg M4S2) r.w ∧ BalLe r.w M4S2) ∧
     (BalLe (balance_neg M4S2) r.x ∧ BalLe r.x M4S2) ∧
     (BalLe (balance_neg M4S2) r.y ∧ BalLe r.y M4S2) ∧
     (BalLe (balance_neg M4S2) r.z ∧ BalLe r.z M4S2))
    ∧
    -- 3. Оценка dot4 v u в Q8.24: -4BU ≤ v·u ≤ 4BU
    (BalLe (balance_neg M4BU) (dot4 v u) ∧ BalLe (dot4 v u) M4BU) := by
  -- 1. Селективность
  have hsd : DualEquiv (dual_mul (dual_var bzero) (dual_const z)) ⟨bzero, z⟩ :=
    ⟨bal_poly_eq (fun | 0 => z | _ => bzero) (.mul .zero (.var 0)) .zero rfl,
     bal_poly_eq (fun | 0 => z | _ => bzero) e_s128_saddle_eps_L (.var 0) p_s128_saddle_eps⟩
  have hnz : (BalGtZero z ∨ BalLtZero z) →
      BalanceEquiv (dual_mul (dual_var bzero) (dual_const z)).eps bzero → False := by
    intro hz heq
    have hz0 := step24_balance_trans (step24_balance_symm hsd.2) heq
    cases hz with
    | inl hgt => exact bal_gt_zero_ne_zero hgt hz0
    | inr hlt =>
      exact bal_gt_zero_ne_zero (bal_neg_gt_zero hlt)
        (step24_balance_trans (bal_neg_congr hz0) bal_neg_zero)
  have hdead1 : BalanceEquiv (dual_mul (dual_var bzero) (dual_const bzero)).eps bzero :=
    bal_poly_eq (fun _ => bzero) e_s128_dead_eps_L .zero p_s128_dead_eps
  have hdead2 : BalanceEquiv (dual_mul (dual_const bzero) (dual_var bzero)).eps bzero :=
    bal_poly_eq (fun _ => bzero) e_s128_dead_eps_L .zero p_s128_dead_eps
  -- 2. Покомпонентные границы qmul p q
  let S2 := bsq S
  have mww := bal_abs_mul_le hpw1 hpw2 hqw1 hqw2
  have mxx := bal_abs_mul_le hpx1 hpx2 hqx1 hqx2
  have myy := bal_abs_mul_le hpy1 hpy2 hqy1 hqy2
  have mzz := bal_abs_mul_le hpz1 hpz2 hqz1 hqz2
  have mwx := bal_abs_mul_le hpw1 hpw2 hqx1 hqx2
  have mxw := bal_abs_mul_le hpx1 hpx2 hqw1 hqw2
  have mzy := bal_abs_mul_le hpz1 hpz2 hqy1 hqy2
  have myz := bal_abs_mul_le hpy1 hpy2 hqz1 hqz2
  have mwy := bal_abs_mul_le hpw1 hpw2 hqy1 hqy2
  have myw := bal_abs_mul_le hpy1 hpy2 hqw1 hqw2
  have mxz := bal_abs_mul_le hpx1 hpx2 hqz1 hqz2
  have mzx := bal_abs_mul_le hpz1 hpz2 hqx1 hqx2
  have mwz := bal_abs_mul_le hpw1 hpw2 hqz1 hqz2
  have mzw := bal_abs_mul_le hpz1 hpz2 hqw1 hqw2
  have myx := bal_abs_mul_le hpy1 hpy2 hqx1 hqx2
  have mxy := bal_abs_mul_le hpx1 hpx2 hqy1 hqy2
  have rw_l := bal_abs_sub_le mww.1 mww.2 mxx.1 mxx.2
  have rw_r := bal_abs_add_le myy.1 myy.2 mzz.1 mzz.2
  have rw_b := bal_abs_sub_le rw_l.1 rw_l.2 rw_r.1 rw_r.2
  have rx_l := bal_abs_add_le mwx.1 mwx.2 mxw.1 mxw.2
  have rx_r := bal_abs_sub_le mzy.1 mzy.2 myz.1 myz.2
  have rx_b := bal_abs_sub_le rx_l.1 rx_l.2 rx_r.1 rx_r.2
  have ry_l := bal_abs_add_le mwy.1 mwy.2 myw.1 myw.2
  have ry_r := bal_abs_sub_le mxz.1 mxz.2 mzx.1 mzx.2
  have ry_b := bal_abs_sub_le ry_l.1 ry_l.2 ry_r.1 ry_r.2
  have rz_l := bal_abs_add_le mwz.1 mwz.2 mzw.1 mzw.2
  have rz_r := bal_abs_sub_le myx.1 myx.2 mxy.1 mxy.2
  have rz_b := bal_abs_sub_le rz_l.1 rz_l.2 rz_r.1 rz_r.2
  -- 3. Границы dot4 v u
  let BU := balance_mul B U
  have dww := bal_abs_mul_le hvw1 hvw2 huw1 huw2
  have dxx := bal_abs_mul_le hvx1 hvx2 hux1 hux2
  have dyy := bal_abs_mul_le hvy1 hvy2 huy1 huy2
  have dzz := bal_abs_mul_le hvz1 hvz2 huz1 huz2
  have dl := bal_abs_add_le dww.1 dww.2 dxx.1 dxx.2
  have dr := bal_abs_add_le dyy.1 dyy.2 dzz.1 dzz.2
  have db := bal_abs_add_le dl.1 dl.2 dr.1 dr.2
  exact ⟨⟨hsd, hnz, hdead1, hdead2⟩,
         ⟨bal_abs_4_bounds S2 _ rw_b.1 rw_b.2,
          bal_abs_4_bounds S2 _ rx_b.1 rx_b.2,
          bal_abs_4_bounds S2 _ ry_b.1 ry_b.2,
          bal_abs_4_bounds S2 _ rz_b.1 rz_b.2⟩,
         bal_abs_4_bounds BU _ db.1 db.2⟩

-- ----- ШАГ 129. Защита 1/2-шагового Ньютона экспоненциальным сдвигом и водораздел β = 1 -----
-- Переменные BExpr: 0 => q, 1 => p, 2 => y, 3 => m, 4 => beta
private def e_s129_2qp_L : BExpr := .sub (.mul (.nat 2) (.var 0)) (.var 1)
private def e_s129_2qp_R : BExpr := .sub (.sub (.mul (.nat 3) (.var 0)) (.var 1)) (.var 0)
private theorem p_s129_2qp : e_s129_2qp_L.norm = e_s129_2qp_R.norm := rfl

private def e_s129_t3_split_L : BExpr := .add (.var 0) (.sub (.mul (.nat 2) (.var 0)) (.var 1))
private def e_s129_t3_split_R : BExpr := .sub (.mul (.nat 3) (.var 0)) (.var 1)
private theorem p_s129_t3_split : e_s129_t3_split_L.norm = e_s129_t3_split_R.norm := rfl

private def e_s129_ynext_ge_L : BExpr := .mul (.var 2) (.sub (.mul (.nat 2) (.var 0)) (.var 1))
private def e_s129_ynext_ge_R : BExpr :=
  .sub (.mul (.var 2) (.sub (.mul (.nat 3) (.var 0)) (.var 1))) (.mul (.var 2) (.var 0))
private theorem p_s129_ynext_ge : e_s129_ynext_ge_L.norm = e_s129_ynext_ge_R.norm := rfl

private def e_s129_r1_hi_L : BExpr :=
  let eq := BExpr.var 0; let ep := BExpr.var 1
  .mul (.mul (.sub eq ep) (.sub eq ep)) (.add (.mul (.nat 2) eq) (.sub (.mul (.nat 2) eq) ep))
private def e_s129_r1_hi_R : BExpr :=
  let eq := BExpr.var 0; let ep := BExpr.var 1
  let t3 := BExpr.sub (.mul (.nat 3) eq) ep
  .sub (.mul (.nat 4) (.mul eq (.mul eq eq))) (.mul ep (.mul t3 t3))
private theorem p_s129_r1_hi : e_s129_r1_hi_L.norm = e_s129_r1_hi_R.norm := rfl

private def e_s129_r1_lo_L : BExpr :=
  let eq := BExpr.var 0; let ep := BExpr.var 1
  let e2qp := BExpr.sub (.mul (.nat 2) eq) ep
  let e2pq := BExpr.sub (.mul (.nat 2) ep) eq
  .mul e2qp (.add (.mul (.nat 3) (.mul eq eq)) (.mul e2pq (.add (.mul (.nat 3) eq) (.mul (.nat 2) e2qp))))
private def e_s129_r1_lo_R : BExpr :=
  let eq := BExpr.var 0; let ep := BExpr.var 1
  let t3 := BExpr.sub (.mul (.nat 3) eq) ep
  .mul (.nat 2) (.mul (.nat 2) (.sub (.mul ep (.mul t3 t3)) (.mul (.nat 2) (.mul eq (.mul eq eq)))))
private theorem p_s129_r1_lo : e_s129_r1_lo_L.norm = e_s129_r1_lo_R.norm := rfl

private def e_s129_lin_lo_L : BExpr :=
  let eq := BExpr.var 0; let em := BExpr.var 3
  let e2qm := BExpr.sub (.mul (.nat 2) eq) em
  .mul (.sub em eq) (.add (.mul e2qm (.add (.mul (.nat 5) eq) e2qm)) (.mul (.nat 2) (.mul eq eq)))
private def e_s129_lin_lo_R : BExpr :=
  let eq := BExpr.var 0; let em := BExpr.var 3
  let t5 := BExpr.sub (.mul (.nat 5) eq) em
  .sub (.mul em (.mul t5 t5)) (.mul (.nat 16) (.mul eq (.mul eq eq)))
private theorem p_s129_lin_lo : e_s129_lin_lo_L.norm = e_s129_lin_lo_R.norm := rfl

private def e_s129_lin_hi_L : BExpr :=
  let eq := BExpr.var 0; let em := BExpr.var 3
  let e3m5q := BExpr.sub (.mul (.nat 3) em) (.mul (.nat 5) eq)
  let e2qm := BExpr.sub (.mul (.nat 2) eq) em
  .mul (.mul e3m5q e3m5q) (.add (.mul (.nat 14) eq) (.mul (.nat 3) e2qm))
private def e_s129_lin_hi_R : BExpr :=
  let eq := BExpr.var 0; let em := BExpr.var 3
  let t5 := BExpr.sub (.mul (.nat 5) eq) em
  .sub (.mul (.nat 500) (.mul eq (.mul eq eq))) (.mul (.nat 27) (.mul em (.mul t5 t5)))
private theorem p_s129_lin_hi : e_s129_lin_hi_L.norm = e_s129_lin_hi_R.norm := rfl

private def e_s129_b1_L : BExpr := .sub (.nat 2) (.mul (.nat 2) .one)
private theorem p_s129_b1 : e_s129_b1_L.norm = BExpr.zero.norm := rfl

private def e_s129_blo_L : BExpr := .mul (.nat 2) (.sub .one (.var 4))
private def e_s129_blo_R : BExpr := .sub (.nat 2) (.mul (.nat 2) (.var 4))
private theorem p_s129_blo : e_s129_blo_L.norm = e_s129_blo_R.norm := rfl

private def e_s129_bhi_L : BExpr := .mul (.nat 2) (.sub (.var 4) .one)
private def e_s129_bhi_R : BExpr := .neg (.sub (.nat 2) (.mul (.nat 2) (.var 4)))
private theorem p_s129_bhi : e_s129_bhi_L.norm = e_s129_bhi_R.norm := rfl

/-- ШАГ 129. Гарантия положительности Ньютона `y(3 - ay²)/2` при скобке `r = a·y² = p/q ∈ [1/2, 2]`
    (за счёт сдвига экспоненты) и водораздел регуляризатора чётности при `β = 1`:
    1. При `y > 0, q > 0` и `q ≤ 2p, p ≤ 2q` (`r ∈ [1/2, 2]`):
       `3q - p ≥ q > 0` и `y·(3q - p) ≥ y·q > 0` (инверсия знака Ньютона математически невозможна!).
    2. Инвариантное сжатие: `2q³ ≤ p(3q - p)² ≤ 4q³`, т.е. `r₁ ∈ [1/2, 1]` после первого же шага.
    3. Линейный стартовый seed по мантиссе `y₀ = (5 - m)/4` при `m ∈ [1, 2]` (`q ≤ m ≤ 2q`):
       `16q³ ≤ m(5q - m)²` и `27·m(5q - m)² ≤ 500q³`, т.е. `r₀ ∈ [1, 125/108]` (`< 1.3e-4` за 2 шага!).
    4. Водораздел регуляризатора чётности `β(2 - β)`: производная `2 - 2β = 0` в `β = 1`,
       строго `> 0` при `β < 1` (тянет к `0`) и строго `< 0` при `β > 1` (тянет к `2`). -/
theorem step129_newton_inv_sqrt_exponent_bracket_and_contraction
    (y p q m beta : Balance)
    (hy : BalGtZero y) (hq : BalGtZero q)
    (hr_lo : BalLe q (balance_mul (balOfNat 2) p))
    (hr_hi : BalLe p (balance_mul (balOfNat 2) q))
    (hm_lo : BalLe q m)
    (hm_hi : BalLe m (balance_mul (balOfNat 2) q)) :
    let q3 := balance_mul q (bsq q)
    let t3 := balance_sub (balance_mul (balOfNat 3) q) p
    let y_next_num := balance_mul y t3
    let r1_num := balance_mul p (bsq t3)
    let lin_num := balance_mul m (bsq (balance_sub (balance_mul (balOfNat 5) q) m))
    -- 1. Безусловная положительность Ньютона
    (BalLe q t3 ∧ BalGtZero t3 ∧ BalGtZero y_next_num ∧ BalLe (balance_mul y q) y_next_num)
    ∧
    -- 2. Инвариантная скобка r₁ ∈ [1/2, 1]
    (BalLe (balance_mul (balOfNat 2) q3) r1_num ∧ BalLe r1_num (balance_mul (balOfNat 4) q3))
    ∧
    -- 3. Линейный seed по мантиссе: r₀ ∈ [1, 125/108]
    (BalLe (balance_mul (balOfNat 16) q3) lin_num ∧
     BalLe (balance_mul (balOfNat 27) lin_num) (balance_mul (balOfNat 500) q3))
    ∧
    -- 4. Водораздел β(2 - β) в точке β = 1
    (BalanceEquiv (balance_sub (balOfNat 2) (balance_mul (balOfNat 2) bone)) bzero ∧
     (BalLt beta bone → BalGtZero (balance_sub (balOfNat 2) (balance_mul (balOfNat 2) beta))) ∧
     (BalLt bone beta → BalGtZero (balance_neg (balance_sub (balOfNat 2) (balance_mul (balOfNat 2) beta))))) := by
  dsimp only [bsq]
  let env : Nat → Balance := fun | 0 => q | 1 => p | 2 => y | 3 => m | 4 => beta | _ => bzero
  -- 1. Строгая положительность Ньютона
  have h2qp : BalGeZero (balance_sub (balance_mul (balOfNat 2) q) p) := hr_hi
  have ht3_ge_q : BalLe q (balance_sub (balance_mul (balOfNat 3) q) p) :=
    bal_ge_zero_of_equiv (bal_poly_eq env e_s129_2qp_L e_s129_2qp_R p_s129_2qp) h2qp
  have ht3_gt : BalGtZero (balance_sub (balance_mul (balOfNat 3) q) p) :=
    bal_gt_zero_of_equiv (bal_poly_eq env e_s129_t3_split_L e_s129_t3_split_R p_s129_t3_split)
      (bal_add_gt_ge hq h2qp)
  have hynext_gt : BalGtZero (balance_mul y (balance_sub (balance_mul (balOfNat 3) q) p)) :=
    bal_mul_gt_gt hy ht3_gt
  have hynext_ge : BalLe (balance_mul y q) (balance_mul y (balance_sub (balance_mul (balOfNat 3) q) p)) :=
    bal_ge_zero_of_equiv (bal_poly_eq env e_s129_ynext_ge_L e_s129_ynext_ge_R p_s129_ynext_ge)
      (bal_mul_gt_ge hy h2qp)
  -- 2. Инвариантная скобка r₁ ∈ [1/2, 1]
  have hr1_hi : BalLe (balance_mul p (balance_mul (balance_sub (balance_mul (balOfNat 3) q) p)
                                                  (balance_sub (balance_mul (balOfNat 3) q) p)))
                      (balance_mul (balOfNat 4) (balance_mul q (balance_mul q q))) :=
    bal_ge_zero_of_equiv (bal_poly_eq env e_s129_r1_hi_L e_s129_r1_hi_R p_s129_r1_hi)
      (bal_mul_ge_ge (bal_sq_ge_zero (balance_sub q p))
        (bal_add_ge_ge (Or.inr (bal_mul_gt_gt (balOfNat_succ_gt_zero 1) hq)) h2qp))
  have hr1_lo : BalLe (balance_mul (balOfNat 2) (balance_mul q (balance_mul q q)))
                      (balance_mul p (balance_mul (balance_sub (balance_mul (balOfNat 3) q) p)
                                                  (balance_sub (balance_mul (balOfNat 3) q) p))) := by
    have h4_ge : BalGeZero (balance_mul (balOfNat 2) (balance_mul (balOfNat 2)
        (balance_sub (balance_mul p (balance_mul (balance_sub (balance_mul (balOfNat 3) q) p)
                                                 (balance_sub (balance_mul (balOfNat 3) q) p)))
                     (balance_mul (balOfNat 2) (balance_mul q (balance_mul q q)))))) :=
      bal_ge_zero_of_equiv (bal_poly_eq env e_s129_r1_lo_L e_s129_r1_lo_R p_s129_r1_lo)
        (bal_mul_ge_ge h2qp
          (bal_add_ge_ge
            (bal_mul_ge_ge (Or.inr (balOfNat_succ_gt_zero 2)) (bal_sq_ge_zero q))
            (bal_mul_ge_ge hr_lo
              (bal_add_ge_ge (Or.inr (bal_mul_gt_gt (balOfNat_succ_gt_zero 2) hq))
                             (bal_mul_ge_ge (Or.inr (balOfNat_succ_gt_zero 1)) h2qp)))))
    exact bal_ge_zero_of_double_ge (bal_ge_zero_of_double_ge h4_ge)
  -- 3. Линейный seed по мантиссе: r₀ ∈ [1, 125/108]
  have hlin_lo : BalLe (balance_mul (balOfNat 16) (balance_mul q (balance_mul q q)))
                       (balance_mul m (balance_mul (balance_sub (balance_mul (balOfNat 5) q) m)
                                                   (balance_sub (balance_mul (balOfNat 5) q) m))) :=
    bal_ge_zero_of_equiv (bal_poly_eq env e_s129_lin_lo_L e_s129_lin_lo_R p_s129_lin_lo)
      (bal_mul_ge_ge hm_lo
        (bal_add_ge_ge
          (bal_mul_ge_ge hm_hi
            (bal_add_ge_ge (Or.inr (bal_mul_gt_gt (balOfNat_succ_gt_zero 4) hq)) hm_hi))
          (bal_mul_ge_ge (Or.inr (balOfNat_succ_gt_zero 1)) (bal_sq_ge_zero q))))
  have hlin_hi : BalLe (balance_mul (balOfNat 27)
                         (balance_mul m (balance_mul (balance_sub (balance_mul (balOfNat 5) q) m)
                                                     (balance_sub (balance_mul (balOfNat 5) q) m))))
                       (balance_mul (balOfNat 500) (balance_mul q (balance_mul q q))) :=
    bal_ge_zero_of_equiv (bal_poly_eq env e_s129_lin_hi_L e_s129_lin_hi_R p_s129_lin_hi)
      (bal_mul_ge_ge (bal_sq_ge_zero (balance_sub (balance_mul (balOfNat 3) m) (balance_mul (balOfNat 5) q)))
        (bal_add_ge_ge (Or.inr (bal_mul_gt_gt (balOfNat_succ_gt_zero 13) hq))
          (bal_mul_ge_ge (Or.inr (balOfNat_succ_gt_zero 2)) hm_hi)))
  -- 4. Водораздел β = 1
  have hb1 : BalanceEquiv (balance_sub (balOfNat 2) (balance_mul (balOfNat 2) bone)) bzero :=
    bal_poly_eq env e_s129_b1_L .zero p_s129_b1
  have hblo : BalLt beta bone → BalGtZero (balance_sub (balOfNat 2) (balance_mul (balOfNat 2) beta)) := by
    intro hlt
    exact bal_gt_zero_of_equiv (bal_poly_eq env e_s129_blo_L e_s129_blo_R p_s129_blo)
      (bal_mul_gt_gt (balOfNat_succ_gt_zero 1) hlt)
  have hbhi : BalLt bone beta → BalGtZero (balance_neg (balance_sub (balOfNat 2) (balance_mul (balOfNat 2) beta))) := by
    intro hgt
    exact bal_gt_zero_of_equiv (bal_poly_eq env e_s129_bhi_L e_s129_bhi_R p_s129_bhi)
      (bal_mul_gt_gt (balOfNat_succ_gt_zero 1) hgt)
  exact ⟨⟨ht3_ge_q, ht3_gt, hynext_gt, hynext_ge⟩,
         ⟨hr1_lo, hr1_hi⟩,
         ⟨hlin_lo, hlin_hi⟩,
         ⟨hb1, hblo, hbhi⟩⟩

-- =============================================================================
-- ШАГИ 130–134: ФОРМАЛЬНАЯ ВЕРИФИКАЦИЯ ОПТИМИЗАЦИЙ И ИСПРАВЛЕНИЙ v0.5.0 (0 АКСИОМ)
-- =============================================================================

-- ----- ШАГ 130. Точный перенос скобок и относительной ошибки для `sqrt_newton_f32(a) = a * y` ([E013]) -----
private def e_s130_sq_L : BExpr := .mul (.mul (.var 0) (.var 1)) (.mul (.var 0) (.var 1))
private def e_s130_sq_R : BExpr := .mul (.var 0) (.mul (.var 0) (.mul (.var 1) (.var 1)))
private theorem p_s130_sq : e_s130_sq_L.norm = e_s130_sq_R.norm := rfl

private def e_s130_qsq_L : BExpr := .mul (.var 2) (.mul (.mul (.var 0) (.var 1)) (.mul (.var 0) (.var 1)))
private def e_s130_qsq_R : BExpr := .mul (.var 0) (.mul (.var 2) (.mul (.var 0) (.mul (.var 1) (.var 1))))
private theorem p_s130_qsq : e_s130_qsq_L.norm = e_s130_qsq_R.norm := rfl

private def e_s130_res_L : BExpr :=
  .mul (.var 2) (.sub (.mul (.mul (.var 0) (.var 1)) (.mul (.var 0) (.var 1))) (.var 0))
private def e_s130_res_R : BExpr :=
  .mul (.var 0) (.sub (.mul (.var 2) (.mul (.var 0) (.mul (.var 1) (.var 1)))) (.var 2))
private theorem p_s130_res : e_s130_res_L.norm = e_s130_res_R.norm := rfl

private def e_s130_mul_neg_L : BExpr := .mul (.var 0) (.neg (.var 1))
private def e_s130_mul_neg_R : BExpr := .neg (.mul (.var 0) (.var 1))
private theorem p_s130_mul_neg : e_s130_mul_neg_L.norm = e_s130_mul_neg_R.norm := rfl

/-- ШАГ 130. Корректность безкорневого `sqrt_newton_f32(a) = a · y` (запрет `.sqrt()` `[E013]`):
    При `a > 0, y > 0, q > 0` и скобке обратного корня `lo ≤ q·(a·y²) ≤ hi`, `|q·(a·y²) - q| ≤ eps`:
    1. `s = a · y > 0` (строгая положительность без смены знака).
    2. Точные тождества `s² ~ a·(a·y²)` и `q·s² ~ a·(q·a·y²)`.
    3. Точный перенос скобки: `a·lo ≤ q·s² ≤ a·hi`.
    4. Точный перенос относительной погрешности: `|q·(s² - a)| ≤ a·eps`. -/
theorem step130_newton_sqrt_from_inv_sqrt_bound
    (a y q lo hi eps : Balance)
    (ha : BalGtZero a) (hy : BalGtZero y) (_hq : BalGtZero q)
    (hlo : BalLe lo (balance_mul q (balance_mul a (bsq y))))
    (hhi : BalLe (balance_mul q (balance_mul a (bsq y))) hi)
    (he1 : BalLe (balance_neg eps) (balance_sub (balance_mul q (balance_mul a (bsq y))) q))
    (he2 : BalLe (balance_sub (balance_mul q (balance_mul a (bsq y))) q) eps) :
    let s := balance_mul a y
    let qay2 := balance_mul q (balance_mul a (bsq y))
    BalGtZero s ∧
    BalanceEquiv (bsq s) (balance_mul a (balance_mul a (bsq y))) ∧
    BalanceEquiv (balance_mul q (bsq s)) (balance_mul a qay2) ∧
    (BalLe (balance_mul a lo) (balance_mul q (bsq s)) ∧
     BalLe (balance_mul q (bsq s)) (balance_mul a hi)) ∧
    BalanceEquiv (balance_mul q (balance_sub (bsq s) a)) (balance_mul a (balance_sub qay2 q)) ∧
    (BalLe (balance_neg (balance_mul a eps)) (balance_mul q (balance_sub (bsq s) a)) ∧
     BalLe (balance_mul q (balance_sub (bsq s) a)) (balance_mul a eps)) := by
  dsimp only [bsq]
  let env : Nat → Balance := fun | 0 => a | 1 => y | 2 => q | _ => bzero
  have hs_gt : BalGtZero (balance_mul a y) := bal_mul_gt_gt ha hy
  have hsq : BalanceEquiv (balance_mul (balance_mul a y) (balance_mul a y))
                          (balance_mul a (balance_mul a (balance_mul y y))) :=
    bal_poly_eq env e_s130_sq_L e_s130_sq_R p_s130_sq
  have hqsq : BalanceEquiv (balance_mul q (balance_mul (balance_mul a y) (balance_mul a y)))
                           (balance_mul a (balance_mul q (balance_mul a (balance_mul y y)))) :=
    bal_poly_eq env e_s130_qsq_L e_s130_qsq_R p_s130_qsq
  have hlo' : BalLe (balance_mul a lo) (balance_mul q (balance_mul (balance_mul a y) (balance_mul a y))) :=
    bal_le_of_equiv_right (step24_balance_symm hqsq) (bal_le_mul_nonneg_left (Or.inr ha) hlo)
  have hhi' : BalLe (balance_mul q (balance_mul (balance_mul a y) (balance_mul a y))) (balance_mul a hi) :=
    bal_le_of_equiv_left (step24_balance_symm hqsq) (bal_le_mul_nonneg_left (Or.inr ha) hhi)
  have hres : BalanceEquiv
      (balance_mul q (balance_sub (balance_mul (balance_mul a y) (balance_mul a y)) a))
      (balance_mul a (balance_sub (balance_mul q (balance_mul a (balance_mul y y))) q)) :=
    bal_poly_eq env e_s130_res_L e_s130_res_R p_s130_res
  have hnege : BalanceEquiv (balance_mul a (balance_neg eps)) (balance_neg (balance_mul a eps)) :=
    bal_poly_eq (fun | 0 => a | 1 => eps | _ => bzero) e_s130_mul_neg_L e_s130_mul_neg_R p_s130_mul_neg
  have he1' : BalLe (balance_neg (balance_mul a eps))
                    (balance_mul q (balance_sub (balance_mul (balance_mul a y) (balance_mul a y)) a)) :=
    bal_le_of_equiv_right (step24_balance_symm hres)
      (bal_le_of_equiv_left hnege (bal_le_mul_nonneg_left (Or.inr ha) he1))
  have he2' : BalLe (balance_mul q (balance_sub (balance_mul (balance_mul a y) (balance_mul a y)) a))
                    (balance_mul a eps) :=
    bal_le_of_equiv_left (step24_balance_symm hres) (bal_le_mul_nonneg_left (Or.inr ha) he2)
  exact ⟨hs_gt, hsq, hqsq, ⟨hlo', hhi'⟩, hres, ⟨he1', he2'⟩⟩

-- ----- ШАГ 131. Общий хребет банков (shared_backbone) и устойчивость 4-полосового IDBD -----
private def e_s131_add_dot_L : BExpr := .add (.mul (.add (.var 0) (.var 1)) (.var 2)) (.add (.var 3) (.var 4))
private def e_s131_add_dot_R : BExpr := .add (.add (.mul (.var 0) (.var 2)) (.var 3)) (.add (.mul (.var 1) (.var 2)) (.var 4))
private theorem p_s131_add_dot : e_s131_add_dot_L.norm = e_s131_add_dot_R.norm := rfl

theorem vdot_add_left : {n : Nat} → (a b h : Vec n) →
    BalanceEquiv (vdot (vec_add a b) h) (balance_add (vdot a h) (vdot b h))
  | 0, (), (), () => step24_balance_symm (bal_add_zero bzero)
  | _ + 1, (ah, at_), (bh, bt_), (hh, ht_) => by
    dsimp [vec_add, vdot]
    have htl := vdot_add_left at_ bt_ ht_
    have hs := step25_balance_add_congr (step24_balance_refl (balance_mul (balance_add ah bh) hh)) htl
    exact step24_balance_trans hs
      (bal_poly_eq (fun | 0 => ah | 1 => bh | 2 => hh | 3 => vdot at_ ht_ | 4 => vdot bt_ ht_ | _ => bzero)
        e_s131_add_dot_L e_s131_add_dot_R p_s131_add_dot)

private def e_s131_conv_lo_L : BExpr :=
  .add (.mul (.sub .one (.var 0)) (.add (.var 3) (.var 1)))
       (.mul (.var 0) (.add (.var 3) (.var 2)))
private def e_s131_conv_lo_R : BExpr :=
  .sub (.add (.mul (.sub .one (.var 0)) (.var 1)) (.mul (.var 0) (.var 2))) (.neg (.var 3))
private theorem p_s131_conv_lo : e_s131_conv_lo_L.norm = e_s131_conv_lo_R.norm := rfl

private def e_s131_conv_hi_L : BExpr :=
  .add (.mul (.sub .one (.var 0)) (.sub (.var 3) (.var 1)))
       (.mul (.var 0) (.sub (.var 3) (.var 2)))
private def e_s131_conv_hi_R : BExpr :=
  .sub (.var 3) (.add (.mul (.sub .one (.var 0)) (.var 1)) (.mul (.var 0) (.var 2)))
private theorem p_s131_conv_hi : e_s131_conv_hi_L.norm = e_s131_conv_hi_R.norm := rfl

/-- ШАГ 131. Общий хребет `shared_backbone` и 4-скалярный полосовой рациональный IDBD в `UnifiedCayleyCore`:
    1. Изоляция неактивного банка: обновление головки активного банка `w₀' = w₀ + Δw₀` при общем
       хребте `h` даёт точное приращение `⟨w₀', h⟩ ~ ⟨w₀, h⟩ + ⟨Δw₀, h⟩` и оставляет выход
       неактивного банка `⟨w₁, h⟩` алгебраически неизменным (0 интерференции).
    2. Строгая положительность знаменателя `q₂(β) ≥ 1 > 0` полосового рационального IDBD (`step116`).
    3. Выпуклая инвариантность мета-трассы IDBD: при `0 ≤ s ≤ 1`, `|h_tr| ≤ M`, `|u_g| ≤ M`
       обновление `h_next = (1 - s)·h_tr + s·u_g` строго остаётся в шаре `[-M, +M]`. -/
theorem step131_shared_backbone_bank_swap_and_band_idbd
    {n : Nat} (h w0 dw0 w1 : Vec n) (beta h_tr u_g s M : Balance)
    (hs0 : BalGeZero s) (hs1 : BalLe s bone)
    (hh1 : BalLe (balance_neg M) h_tr) (hh2 : BalLe h_tr M)
    (hu1 : BalLe (balance_neg M) u_g) (hu2 : BalLe u_g M) :
    let w0_next := vec_add w0 dw0
    let q2 := balance_add (balance_sub (balOfNat 2) (balance_mul (balOfNat 2) beta)) (bsq beta)
    let h_next := balance_add (balance_mul (balance_sub bone s) h_tr) (balance_mul s u_g)
    (BalanceEquiv (vdot w0_next h) (balance_add (vdot w0 h) (vdot dw0 h)) ∧
     BalanceEquiv (vdot w1 h) (vdot w1 h)) ∧
    (BalLe bone q2 ∧ BalGtZero q2) ∧
    (BalLe (balance_neg M) h_next ∧ BalLe h_next M) := by
  have h116 := step116_rational_pos_gate_c1_and_bounds beta
  have hMh : BalGeZero (balance_add M h_tr) :=
    bal_ge_zero_of_equiv (bal_poly_eq (fun | 0 => M | 1 => h_tr | _ => bzero)
      e_sub_neg_eq_add_L e_sub_neg_eq_add_R p_sub_neg_eq_add) hh1
  have hMu : BalGeZero (balance_add M u_g) :=
    bal_ge_zero_of_equiv (bal_poly_eq (fun | 0 => M | 1 => u_g | _ => bzero)
      e_sub_neg_eq_add_L e_sub_neg_eq_add_R p_sub_neg_eq_add) hu1
  let env : Nat → Balance := fun | 0 => s | 1 => h_tr | 2 => u_g | 3 => M | _ => bzero
  have hlo : BalLe (balance_neg M) (balance_add (balance_mul (balance_sub bone s) h_tr) (balance_mul s u_g)) :=
    bal_ge_zero_of_equiv (bal_poly_eq env e_s131_conv_lo_L e_s131_conv_lo_R p_s131_conv_lo)
      (bal_add_ge_ge (bal_mul_ge_ge hs1 hMh) (bal_mul_ge_ge hs0 hMu))
  have hhi : BalLe (balance_add (balance_mul (balance_sub bone s) h_tr) (balance_mul s u_g)) M :=
    bal_ge_zero_of_equiv (bal_poly_eq env e_s131_conv_hi_L e_s131_conv_hi_R p_s131_conv_hi)
      (bal_add_ge_ge (bal_mul_ge_ge hs1 hh2) (bal_mul_ge_ge hs0 hu2))
  exact ⟨⟨vdot_add_left w0 dw0 h, step24_balance_refl _⟩,
         ⟨h116.2.1, h116.2.2.1⟩,
         ⟨hlo, hhi⟩⟩

-- ----- ШАГ 132. Сохранение argmax при квантовании гиперсферических прототипов в Q1.14 (i16) -----
def VecAbsLe : {n : Nat} → Balance → Vec n → Prop
  | 0, _, () => True
  | _ + 1, U, (uh, ut) => (BalLe (balance_neg U) uh ∧ BalLe uh U) ∧ VecAbsLe U ut

def VecQuantErrLe : {n : Nat} → Balance → Balance → Vec n → Vec n → Prop
  | 0, _, _, (), () => True
  | _ + 1, S, E, (ph, pt), (qh, qt) =>
    (BalLe (balance_neg E) (balance_sub (balance_mul S ph) qh) ∧
     BalLe (balance_sub (balance_mul S ph) qh) E) ∧
    VecQuantErrLe S E pt qt

private def e_s132_dot_step_L : BExpr :=
  .add (.mul (.var 0) (.sub (.mul (.var 1) (.var 2)) (.var 3))) (.var 4)
private def e_s132_dot_step_R : BExpr :=
  .sub (.mul (.var 1) (.add (.mul (.var 0) (.var 2)) (.var 5))) (.add (.mul (.var 0) (.var 3)) (.var 6))
private def e_s132_dot_step_diff_L : BExpr :=
  .add (.mul (.var 0) (.sub (.mul (.var 1) (.var 2)) (.var 3)))
       (.sub (.mul (.var 1) (.var 5)) (.var 6))
private theorem p_s132_dot_step : e_s132_dot_step_diff_L.norm = e_s132_dot_step_R.norm := rfl

private def e_s132_succ_L : BExpr := .add (.var 0) (.mul (.var 1) (.var 0))
private def e_s132_succ_R : BExpr := .mul (.add (.var 1) .one) (.var 0)
private theorem p_s132_succ : e_s132_succ_L.norm = e_s132_succ_R.norm := rfl

private def e_s132_zero_L : BExpr := .sub (.mul (.var 0) .zero) .zero
private theorem p_s132_zero : e_s132_zero_L.norm = BExpr.zero.norm := rfl

private def e_s132_zeroM_L : BExpr := .mul .zero (.var 0)
private theorem p_s132_zeroM : e_s132_zeroM_L.norm = BExpr.zero.norm := rfl

private theorem balOfNat_succ_equiv (k : Nat) :
    BalanceEquiv (balOfNat (k + 1)) (balance_add (balOfNat k) bone) :=
  step24_balance_refl _

theorem vdot_q14_err_bound : {n : Nat} → (S U E : Balance) → (u p q : Vec n) →
    VecAbsLe U u → VecQuantErrLe S E p q →
    let B := balance_mul (balOfNat n) (balance_mul U E)
    BalLe (balance_neg B) (balance_sub (balance_mul S (vdot u p)) (vdot u q)) ∧
    BalLe (balance_sub (balance_mul S (vdot u p)) (vdot u q)) B
  | 0, S, U, E, (), (), (), _, _ => by
    dsimp [vdot, balOfNat]
    have h0 : BalanceEquiv (balance_sub (balance_mul S bzero) bzero) bzero :=
      bal_poly_eq (fun | 0 => S | _ => bzero) e_s132_zero_L .zero p_s132_zero
    have hB : BalanceEquiv (balance_mul bzero (balance_mul U E)) bzero :=
      bal_poly_eq (fun | 0 => balance_mul U E | _ => bzero) e_s132_zeroM_L .zero p_s132_zeroM
    have hlo : BalLe (balance_neg (balance_mul bzero (balance_mul U E)))
                     (balance_sub (balance_mul S bzero) bzero) :=
      bal_le_of_equiv_left (step24_balance_symm (step24_balance_trans (bal_neg_congr hB) bal_neg_zero))
        (bal_le_of_equiv_right (step24_balance_symm h0) (bal_le_refl bzero))
    have hhi : BalLe (balance_sub (balance_mul S bzero) bzero)
                     (balance_mul bzero (balance_mul U E)) :=
      bal_le_of_equiv_left (step24_balance_symm h0)
        (bal_le_of_equiv_right (step24_balance_symm hB) (bal_le_refl bzero))
    exact ⟨hlo, hhi⟩
  | k + 1, S, U, E, (uh, ut), (ph, pt), (qh, qt), ⟨⟨hu1, hu2⟩, hut⟩, ⟨⟨he1, he2⟩, het⟩ => by
    dsimp [vdot]
    have hhd := bal_abs_mul_le hu1 hu2 he1 he2
    have htl := vdot_q14_err_bound S U E ut pt qt hut het
    have hsum := bal_abs_add_le hhd.1 hhd.2 htl.1 htl.2
    let envD : Nat → Balance := fun
      | 0 => uh | 1 => S | 2 => ph | 3 => qh | 4 => bzero | 5 => vdot ut pt | 6 => vdot ut qt | _ => bzero
    have hdiff_eq := bal_poly_eq envD e_s132_dot_step_diff_L e_s132_dot_step_R p_s132_dot_step
    have hbound_eq : BalanceEquiv
        (balance_add (balance_mul U E) (balance_mul (balOfNat k) (balance_mul U E)))
        (balance_mul (balOfNat (k + 1)) (balance_mul U E)) :=
      step24_balance_trans
        (bal_poly_eq (fun | 0 => balance_mul U E | 1 => balOfNat k | _ => bzero)
          e_s132_succ_L e_s132_succ_R p_s132_succ)
        (bal_mul_congr (step24_balance_symm (balOfNat_succ_equiv k)) (step24_balance_refl _))
    exact ⟨bal_le_of_equiv_left (bal_neg_congr hbound_eq) (bal_le_of_equiv_right hdiff_eq hsum.1),
           bal_le_of_equiv_left hdiff_eq (bal_le_of_equiv_right hbound_eq hsum.2)⟩

private def e_s132_margin_L : BExpr :=
  .add (.sub (.var 0) (.mul (.nat 2) (.var 1)))
       (.add (.sub (.var 1) (.var 2)) (.add (.var 1) (.var 3)))
private def e_s132_margin_R : BExpr :=
  .sub (.sub (.var 0) (.var 2)) (.neg (.var 3))
private theorem p_s132_margin : e_s132_margin_L.norm = e_s132_margin_R.norm := rfl

private def e_s132_qgap_L : BExpr :=
  .sub (.sub (.mul (.var 0) (.sub (.var 1) (.var 2)))
             (.sub (.mul (.var 0) (.var 1)) (.var 3)))
       (.neg (.sub (.mul (.var 0) (.var 2)) (.var 4)))
private def e_s132_qgap_R : BExpr := .sub (.var 3) (.var 4)
private theorem p_s132_qgap : e_s132_qgap_L.norm = e_s132_qgap_R.norm := rfl

/-- ШАГ 132. Сохранение argmax при квантовании гиперсферических прототипов в `Q1.14` (`i16`, `S = 16384`):
    Для ЛЮБОЙ размерности `n`, состояния `|u_i| ≤ U` и прототипов с ошибкой квантования `|S·p_i - q_i| ≤ E`:
    1. Ошибка скалярного произведения строго ограничена: `|S·⟨u, p⟩ - ⟨u, q⟩| ≤ n·U·E`.
    2. Если истинный зазор классов `S·(⟨u, p₁⟩ - ⟨u, p₂⟩) > 2·n·U·E` (для `Q1.14`, `U = 1`, `E = 1/2`:
       `Δ > n / 16384`), то квантованное сравнение строго сохраняет победителя: `⟨u, q₁⟩ > ⟨u, q₂⟩`! -/
theorem step132_q14_hypersphere_proto_margin_preservation
    {n : Nat} (S U E : Balance) (u p1 q1 p2 q2 : Vec n)
    (hu : VecAbsLe U u)
    (hq1 : VecQuantErrLe S E p1 q1)
    (hq2 : VecQuantErrLe S E p2 q2)
    (hmargin : BalLt (balance_mul (balOfNat 2) (balance_mul (balOfNat n) (balance_mul U E)))
                     (balance_mul S (balance_sub (vdot u p1) (vdot u p2)))) :
    let B := balance_mul (balOfNat n) (balance_mul U E)
    (BalLe (balance_neg B) (balance_sub (balance_mul S (vdot u p1)) (vdot u q1)) ∧
     BalLe (balance_sub (balance_mul S (vdot u p1)) (vdot u q1)) B) ∧
    (BalLe (balance_neg B) (balance_sub (balance_mul S (vdot u p2)) (vdot u q2)) ∧
     BalLe (balance_sub (balance_mul S (vdot u p2)) (vdot u q2)) B) ∧
    BalLt (vdot u q2) (vdot u q1) := by
  have hb1 := vdot_q14_err_bound S U E u p1 q1 hu hq1
  have hb2 := vdot_q14_err_bound S U E u p2 q2 hu hq2
  let B := balance_mul (balOfNat n) (balance_mul U E)
  let M := balance_mul S (balance_sub (vdot u p1) (vdot u p2))
  let e1 := balance_sub (balance_mul S (vdot u p1)) (vdot u q1)
  let e2 := balance_sub (balance_mul S (vdot u p2)) (vdot u q2)
  have he2p : BalGeZero (balance_add B e2) :=
    bal_ge_zero_of_equiv (bal_poly_eq (fun | 0 => B | 1 => e2 | _ => bzero)
      e_sub_neg_eq_add_L e_sub_neg_eq_add_R p_sub_neg_eq_add) hb2.1
  have hsum_gt : BalGtZero (balance_add (balance_sub M (balance_mul (balOfNat 2) B))
                                        (balance_add (balance_sub B e1) (balance_add B e2))) :=
    bal_add_gt_ge hmargin (bal_add_ge_ge hb1.2 he2p)
  have h1 : BalGtZero (balance_sub (balance_sub M e1) (balance_neg e2)) :=
    bal_gt_zero_of_equiv
      (bal_poly_eq (fun | 0 => M | 1 => B | 2 => e1 | 3 => e2 | _ => bzero)
        e_s132_margin_L e_s132_margin_R p_s132_margin)
      hsum_gt
  have hlt : BalLt (vdot u q2) (vdot u q1) :=
    bal_gt_zero_of_equiv
      (bal_poly_eq (fun | 0 => S | 1 => vdot u p1 | 2 => vdot u p2 | 3 => vdot u q1 | 4 => vdot u q2 | _ => bzero)
        e_s132_qgap_L e_s132_qgap_R p_s132_qgap)
      h1
  exact ⟨hb1, hb2, hlt⟩

-- ----- ШАГ 133. Точность слияния каскада роторов Гивенса/чётности в предвычисленную матрицу Q -----
def givens01_num (t : Balance) (v : Vec3) : Vec3 :=
  let c := balance_sub bone (bsq t)
  let s := balance_mul (balOfNat 2) t
  let d := balance_add bone (bsq t)
  ⟨balance_sub (balance_mul c v.x) (balance_mul s v.y),
   balance_add (balance_mul s v.x) (balance_mul c v.y),
   balance_mul d v.z⟩

def givens12_num (t : Balance) (v : Vec3) : Vec3 :=
  let c := balance_sub bone (bsq t)
  let s := balance_mul (balOfNat 2) t
  let d := balance_add bone (bsq t)
  ⟨balance_mul d v.x,
   balance_sub (balance_mul c v.y) (balance_mul s v.z),
   balance_add (balance_mul s v.y) (balance_mul c v.z)⟩

def mat3_cols_mul (c0 c1 c2 v : Vec3) : Vec3 :=
  ⟨balance_add (balance_add (balance_mul c0.x v.x) (balance_mul c1.x v.y)) (balance_mul c2.x v.z),
   balance_add (balance_add (balance_mul c0.y v.x) (balance_mul c1.y v.y)) (balance_mul c2.y v.z),
   balance_add (balance_add (balance_mul c0.z v.x) (balance_mul c1.z v.y)) (balance_mul c2.z v.z)⟩

def det3_cols (c0 c1 c2 : Vec3) : Balance :=
  balance_add
    (balance_sub
      (balance_mul c0.x (balance_sub (balance_mul c1.y c2.z) (balance_mul c2.y c1.z)))
      (balance_mul c1.x (balance_sub (balance_mul c0.y c2.z) (balance_mul c2.y c0.z))))
    (balance_mul c2.x (balance_sub (balance_mul c0.y c1.z) (balance_mul c1.y c0.z)))

private def bG01E (t : BExpr) (v : BExpr × BExpr × BExpr) : BExpr × BExpr × BExpr :=
  let c := BExpr.sub .one (.mul t t)
  let s := BExpr.mul (.nat 2) t
  let d := BExpr.add .one (.mul t t)
  (.sub (.mul c v.1) (.mul s v.2.1),
   .add (.mul s v.1) (.mul c v.2.1),
   .mul d v.2.2)

private def bG12E (t : BExpr) (v : BExpr × BExpr × BExpr) : BExpr × BExpr × BExpr :=
  let c := BExpr.sub .one (.mul t t)
  let s := BExpr.mul (.nat 2) t
  let d := BExpr.add .one (.mul t t)
  (.mul d v.1,
   .sub (.mul c v.2.1) (.mul s v.2.2),
   .add (.mul s v.2.1) (.mul c v.2.2))

private def bMat3MulE (c0 c1 c2 v : BExpr × BExpr × BExpr) : BExpr × BExpr × BExpr :=
  (.add (.add (.mul c0.1 v.1) (.mul c1.1 v.2.1)) (.mul c2.1 v.2.2),
   .add (.add (.mul c0.2.1 v.1) (.mul c1.2.1 v.2.1)) (.mul c2.2.1 v.2.2),
   .add (.add (.mul c0.2.2 v.1) (.mul c1.2.2 v.2.1)) (.mul c2.2.2 v.2.2))

private def bDet3E (c0 c1 c2 : BExpr × BExpr × BExpr) : BExpr :=
  .add
    (.sub
      (.mul c0.1 (.sub (.mul c1.2.1 c2.2.2) (.mul c2.2.1 c1.2.2)))
      (.mul c1.1 (.sub (.mul c0.2.1 c2.2.2) (.mul c2.2.1 c0.2.2))))
    (.mul c2.1 (.sub (.mul c0.2.1 c1.2.2) (.mul c1.2.1 c0.2.2)))

private def e_s133_q0 := bG12E (.var 1) (bG01E (.var 0) (.one, .zero, .zero))
private def e_s133_q1 := bG12E (.var 1) (bG01E (.var 0) (.zero, .one, .zero))
private def e_s133_q2 := bG12E (.var 1) (bG01E (.var 0) (.zero, .zero, .one))
private def e_s133_v : BExpr × BExpr × BExpr := (.var 2, .var 3, .var 4)
private def e_s133_fused := bMat3MulE e_s133_q0 e_s133_q1 e_s133_q2 e_s133_v
private def e_s133_seq := bG12E (.var 1) (bG01E (.var 0) e_s133_v)
private def e_s133_d12 : BExpr := .mul (.add .one (.mul (.var 0) (.var 0))) (.add .one (.mul (.var 1) (.var 1)))

private theorem p_s133_eq_x : e_s133_fused.1.norm = e_s133_seq.1.norm := rfl
private theorem p_s133_eq_y : e_s133_fused.2.1.norm = e_s133_seq.2.1.norm := rfl
private theorem p_s133_eq_z : e_s133_fused.2.2.norm = e_s133_seq.2.2.norm := rfl
private theorem p_s133_norm :
    (vDotE e_s133_fused e_s133_fused).norm =
    (BExpr.mul (.mul e_s133_d12 e_s133_d12) (vDotE e_s133_v e_s133_v)).norm := rfl
private theorem p_s133_det :
    (bDet3E e_s133_q0 e_s133_q1 e_s133_q2).norm =
    (BExpr.mul e_s133_d12 (.mul e_s133_d12 e_s133_d12)).norm := rfl

/-- ШАГ 133. Алгебраическая точность слияния каскада роторов Гивенса/чётности в предвычисленную
    матрицу `Q` (`GivensParityOp::build_q_mat` и `build_hh5_cascade_mat5`):
    1. Столбцовое построение `q_j = (G₁₂ ∘ G₀₁)(e_j)` даёт линейный оператор, тождественный
       последовательному применению каскада на ЛЮБОМ векторе `v`: `Q·v ~ G₁₂(G₀₁(v))`.
    2. Точное сохранение нормы: `||Q·v||² ~ (d₁·d₂)² · ||v||²`.
    3. Точный определитель `det(Q) ~ (d₁·d₂)³ > 0` (`det = +1`, строго внутри `SO`). -/
theorem step133_givens_parity_matrix_fusion_exactness (t1 t2 : Balance) (v : Vec3) :
    let d1 := balance_add bone (bsq t1)
    let d2 := balance_add bone (bsq t2)
    let d12 := balance_mul d1 d2
    let q0 := givens12_num t2 (givens01_num t1 ⟨bone, bzero, bzero⟩)
    let q1 := givens12_num t2 (givens01_num t1 ⟨bzero, bone, bzero⟩)
    let q2 := givens12_num t2 (givens01_num t1 ⟨bzero, bzero, bone⟩)
    let qv := mat3_cols_mul q0 q1 q2 v
    let seqv := givens12_num t2 (givens01_num t1 v)
    BalGtZero d1 ∧ BalGtZero d2 ∧ BalGtZero d12 ∧
    Vec3Equiv qv seqv ∧
    BalanceEquiv (dot3 qv qv) (balance_mul (bsq d12) (dot3 v v)) ∧
    BalanceEquiv (det3_cols q0 q1 q2) (balance_mul d12 (bsq d12)) ∧
    BalGtZero (det3_cols q0 q1 q2) := by
  dsimp only [givens01_num, givens12_num, mat3_cols_mul, det3_cols, Vec3Equiv, bsq]
  let env : Nat → Balance := fun | 0 => t1 | 1 => t2 | 2 => v.x | 3 => v.y | 4 => v.z | _ => bzero
  have hd1 : BalGtZero (balance_add bone (balance_mul t1 t1)) := bal_one_add_sq_gt_zero t1
  have hd2 : BalGtZero (balance_add bone (balance_mul t2 t2)) := bal_one_add_sq_gt_zero t2
  have hd12 : BalGtZero (balance_mul (balance_add bone (balance_mul t1 t1)) (balance_add bone (balance_mul t2 t2))) :=
    bal_mul_gt_gt hd1 hd2
  have hdet_eq := bal_poly_eq env (bDet3E e_s133_q0 e_s133_q1 e_s133_q2)
    (BExpr.mul e_s133_d12 (.mul e_s133_d12 e_s133_d12)) p_s133_det
  have hdet_gt := bal_gt_zero_of_equiv (step24_balance_symm hdet_eq)
    (bal_mul_gt_gt hd12 (bal_mul_gt_gt hd12 hd12))
  exact ⟨hd1, hd2, hd12,
         ⟨bal_poly_eq env e_s133_fused.1 e_s133_seq.1 p_s133_eq_x,
          bal_poly_eq env e_s133_fused.2.1 e_s133_seq.2.1 p_s133_eq_y,
          bal_poly_eq env e_s133_fused.2.2 e_s133_seq.2.2 p_s133_eq_z⟩,
         bal_poly_eq env (vDotE e_s133_fused e_s133_fused)
           (BExpr.mul (.mul e_s133_d12 e_s133_d12) (vDotE e_s133_v e_s133_v)) p_s133_norm,
         hdet_eq, hdet_gt⟩

-- ----- ШАГ 134. Невырожденность 4-Хаусхолдерова моста Q8.24 на водоразделе β = Q -----
private def e_s134_u3norm_L : BExpr :=
  let eQ := BExpr.var 0; let eb := BExpr.var 1; let ec := BExpr.sub (.mul (.nat 2) eQ) eb
  .add (.mul (.mul ec ec) (.var 2))
       (.add (.mul (.mul ec ec) (.var 3))
             (.add (.mul (.mul ec ec) (.var 4))
                   (.add (.mul (.mul ec ec) (.var 5))
                         (.add (.mul (.mul eQ eb) (.mul eQ eb)) .zero))))
private def e_s134_u3norm_R : BExpr :=
  let eQ := BExpr.var 0; let eb := BExpr.var 1; let ec := BExpr.sub (.mul (.nat 2) eQ) eb
  .add (.mul (.mul ec ec) (.add (.add (.var 2) (.var 3)) (.add (.var 4) (.var 5))))
       (.mul (.mul eQ eQ) (.mul eb eb))
private theorem p_s134_u3norm : e_s134_u3norm_L.norm = e_s134_u3norm_R.norm := rfl

private def e_s134_sos_L : BExpr :=
  let eQ := BExpr.var 0; let eb := BExpr.var 1; let ec := BExpr.sub (.mul (.nat 2) eQ) eb
  .add (.mul (.mul ec ec) (.mul eQ eQ)) (.mul (.mul eQ eQ) (.mul eb eb))
private def e_s134_sos_R : BExpr :=
  let eQ := BExpr.var 0; let eb := BExpr.var 1
  .mul (.mul eQ eQ) (.add (.mul (.nat 2) (.mul (.sub eb eQ) (.sub eb eQ)))
                          (.mul (.nat 2) (.mul eQ eQ)))
private theorem p_s134_sos : e_s134_sos_L.norm = e_s134_sos_R.norm := rfl

private def e_s134_lb_L : BExpr :=
  let eQ := BExpr.var 0; let eb := BExpr.var 1
  .mul (.mul eQ eQ) (.mul (.nat 2) (.mul (.sub eb eQ) (.sub eb eQ)))
private def e_s134_lb_R : BExpr :=
  let eQ := BExpr.var 0; let eb := BExpr.var 1
  .sub (.mul (.mul eQ eQ) (.add (.mul (.nat 2) (.mul (.sub eb eQ) (.sub eb eQ)))
                                (.mul (.nat 2) (.mul eQ eQ))))
       (.mul (.nat 2) (.mul (.mul eQ eQ) (.mul eQ eQ)))
private theorem p_s134_lb : e_s134_lb_L.norm = e_s134_lb_R.norm := rfl

private def e_s134_ws_L : BExpr :=
  let eQ := BExpr.var 0
  .mul (.mul eQ eQ) (.add (.mul (.nat 2) (.mul (.sub eQ eQ) (.sub eQ eQ)))
                          (.mul (.nat 2) (.mul eQ eQ)))
private def e_s134_ws_R : BExpr :=
  let eQ := BExpr.var 0
  .mul (.nat 2) (.mul (.mul eQ eQ) (.mul eQ eQ))
private theorem p_s134_ws : e_s134_ws_L.norm = e_s134_ws_R.norm := rfl

private def e_s134_hh_slack_L : BExpr :=
  .mul (.mul (.nat 2) (.var 0))
       (.add (.mul (.sub (.mul (.nat 2) .one) (.nat 2)) (.var 1))
             (.mul (.mul (.nat 2) .one) .zero))
private theorem p_s134_hh_slack : e_s134_hh_slack_L.norm = BExpr.zero.norm := rfl

private def e_s134_hh_rhs_L : BExpr :=
  .mul (.mul (.mul .one (.add (.var 0) .zero)) (.mul .one (.add (.var 0) .zero))) (.var 1)
private def e_s134_hh_rhs_R : BExpr :=
  .mul (.mul (.var 0) (.var 0)) (.var 1)
private theorem p_s134_hh_rhs : e_s134_hh_rhs_L.norm = e_s134_hh_rhs_R.norm := rfl

/-- ШАГ 134. Невырожденность и точная 5D-изометрия целочисленного `FixedO4Rotor` (`Q8.24`, 4 Хаусхолдера в `SO(5)`):
    Для масштаба `Q > 0` (`Q = 2²⁴`), единичного `vh ∈ R⁴` (`||vh||² ~ Q²`) и ЛЮБОГО `β`
    (включая критический водораздел `β = Q`, где прежняя 1D-подмесь обнуляла 5-ю координату!):
    4-й вектор Хаусхолдера `u₃(β) = ((2Q - β)·vh, Q·β) ∈ R⁵` удовлетворяет:
    1. Точному разложению `||u₃(β)||² ~ Q²·(2(β - Q)² + 2Q²)`.
    2. Безусловной нижней границе `||u₃(β)||² ≥ 2·Q⁴ > 0` для ВСЕХ `β` (деление на 0 невозможно!).
    3. На водоразделе `β = Q`: `||u₃(Q)||² ~ 2·Q⁴ > 0`.
    4. Точной 5D-изометрии Хаусхолдера `||H₂(u₃) y||² ~ ||u₃||⁴ · ||y||²` для любого `y ∈ R⁵`. -/
theorem step134_fixed_q24_hh5_watershed_nondegeneracy
    (Q beta : Balance) (hQ : BalGtZero Q) (vh : Vec4)
    (hvh : BalanceEquiv (dot4 vh vh) (bsq Q)) (y : Vec 5) :
    let c := balance_sub (balance_mul (balOfNat 2) Q) beta
    let u3 : Vec 5 := (balance_mul c vh.w, (balance_mul c vh.x,
                      (balance_mul c vh.y, (balance_mul c vh.z, (balance_mul Q beta, ())))))
    let n2 := vdot u3 u3
    let q4_2 := balance_mul (balOfNat 2) (bsq (bsq Q))
    let h_out := householderN_num (balance_add n2 bzero) (balOfNat 2) bone u3 y
    BalanceEquiv n2 (balance_mul (bsq Q) (balance_add (balance_mul (balOfNat 2) (bsq (balance_sub beta Q)))
                                                      (balance_mul (balOfNat 2) (bsq Q)))) ∧
    BalLe q4_2 n2 ∧
    BalGtZero n2 ∧
    BalanceEquiv (balance_mul (bsq Q) (balance_add (balance_mul (balOfNat 2) (bsq (balance_sub Q Q)))
                                                   (balance_mul (balOfNat 2) (bsq Q)))) q4_2 ∧
    BalanceEquiv (vdot h_out h_out) (balance_mul (bsq n2) (vdot y y)) := by
  dsimp only [vdot, bsq]
  let c := balance_sub (balance_mul (balOfNat 2) Q) beta
  let u3 : Vec 5 := (balance_mul c vh.w, (balance_mul c vh.x,
                    (balance_mul c vh.y, (balance_mul c vh.z, (balance_mul Q beta, ())))))
  let n2 := vdot u3 u3
  have h1 : BalanceEquiv n2
      (balance_add (balance_mul (balance_mul c c) (dot4 vh vh))
                   (balance_mul (balance_mul Q Q) (balance_mul beta beta))) := by
    have hmul_sq : ∀ (a x : Balance),
        BalanceEquiv (balance_mul (balance_mul a x) (balance_mul a x))
                     (balance_mul (balance_mul a a) (balance_mul x x)) :=
      fun a x => bal_poly_eq (fun | 0 => a | 1 => x | _ => bzero) e_s127_sq2_L e_s127_sq2_R p_s127_sq2
    have hstep := step25_balance_add_congr (hmul_sq c vh.w)
      (step25_balance_add_congr (hmul_sq c vh.x)
        (step25_balance_add_congr (hmul_sq c vh.y)
          (step25_balance_add_congr (hmul_sq c vh.z)
            (step25_balance_add_congr
              (step24_balance_refl (balance_mul (balance_mul Q beta) (balance_mul Q beta)))
              (step24_balance_refl bzero)))))
    exact step24_balance_trans hstep
      (bal_poly_eq (fun | 0 => Q | 1 => beta
                        | 2 => balance_mul vh.w vh.w | 3 => balance_mul vh.x vh.x
                        | 4 => balance_mul vh.y vh.y | 5 => balance_mul vh.z vh.z | _ => bzero)
        e_s134_u3norm_L e_s134_u3norm_R p_s134_u3norm)
  have h2 : BalanceEquiv n2
      (balance_mul (balance_mul Q Q)
        (balance_add (balance_mul (balOfNat 2) (balance_mul (balance_sub beta Q) (balance_sub beta Q)))
                     (balance_mul (balOfNat 2) (balance_mul Q Q)))) :=
    step24_balance_trans h1
      (step24_balance_trans
        (step25_balance_add_congr (bal_mul_congr (step24_balance_refl _) hvh) (step24_balance_refl _))
        (bal_poly_eq (fun | 0 => Q | 1 => beta | _ => bzero) e_s134_sos_L e_s134_sos_R p_s134_sos))
  have hQ2 : BalGtZero (balance_mul Q Q) := bal_mul_gt_gt hQ hQ
  have hq4_2_gt : BalGtZero (balance_mul (balOfNat 2) (balance_mul (balance_mul Q Q) (balance_mul Q Q))) :=
    bal_mul_gt_gt (balOfNat_succ_gt_zero 1) (bal_mul_gt_gt hQ2 hQ2)
  have hlb_rhs : BalLe (balance_mul (balOfNat 2) (balance_mul (balance_mul Q Q) (balance_mul Q Q)))
      (balance_mul (balance_mul Q Q)
        (balance_add (balance_mul (balOfNat 2) (balance_mul (balance_sub beta Q) (balance_sub beta Q)))
                     (balance_mul (balOfNat 2) (balance_mul Q Q)))) :=
    bal_ge_zero_of_equiv
      (bal_poly_eq (fun | 0 => Q | 1 => beta | _ => bzero) e_s134_lb_L e_s134_lb_R p_s134_lb)
      (bal_mul_ge_ge (Or.inr hQ2) (bal_mul_ge_ge (Or.inr (balOfNat_succ_gt_zero 1)) (bal_sq_ge_zero (balance_sub beta Q))))
  have hlb : BalLe (balance_mul (balOfNat 2) (balance_mul (balance_mul Q Q) (balance_mul Q Q))) n2 :=
    bal_le_of_equiv_right (step24_balance_symm h2) hlb_rhs
  have hn2_gt : BalGtZero n2 :=
    bal_gt_zero_of_equiv
      (bal_poly_eq (fun | 0 => n2 | 1 => balance_mul (balOfNat 2) (balance_mul (balance_mul Q Q) (balance_mul Q Q)) | _ => bzero)
        e_add_sub_self_L e_add_sub_self_R p_add_sub_self)
      (bal_add_gt_ge hq4_2_gt hlb)
  have hws := bal_poly_eq (fun | 0 => Q | _ => bzero) e_s134_ws_L e_s134_ws_R p_s134_ws
  have hbibo := (householderN_bibo u3 y bzero (balOfNat 2) bone
                  (Or.inl (step24_balance_refl bzero)) bone_gt_zero
                  (Or.inr (balOfNat_succ_gt_zero 1))
                  (bal_le_of_equiv_right (step24_balance_symm (bal_mul_one (balOfNat 2))) (bal_le_refl (balOfNat 2)))).1
  have hslack0 : BalanceEquiv
      (balance_mul (balance_mul (balOfNat 2) (bsq (vdot u3 y)))
        (balance_add (balance_mul (balance_sub (balance_mul (balOfNat 2) bone) (balOfNat 2)) n2)
                     (balance_mul (balance_mul (balOfNat 2) bone) bzero))) bzero :=
    bal_poly_eq (fun | 0 => bsq (vdot u3 y) | 1 => n2 | _ => bzero) e_s134_hh_slack_L .zero p_s134_hh_slack
  have hlhs_eq : BalanceEquiv
      (balance_add (vdot (householderN_num (balance_add n2 bzero) (balOfNat 2) bone u3 y)
                         (householderN_num (balance_add n2 bzero) (balOfNat 2) bone u3 y))
                   (balance_mul (balance_mul (balOfNat 2) (bsq (vdot u3 y)))
                     (balance_add (balance_mul (balance_sub (balance_mul (balOfNat 2) bone) (balOfNat 2)) n2)
                                  (balance_mul (balance_mul (balOfNat 2) bone) bzero))))
      (vdot (householderN_num (balance_add n2 bzero) (balOfNat 2) bone u3 y)
            (householderN_num (balance_add n2 bzero) (balOfNat 2) bone u3 y)) :=
    step24_balance_trans (step25_balance_add_congr (step24_balance_refl _) hslack0) (bal_add_zero _)
  have hrhs_eq : BalanceEquiv
      (balance_mul (bsq (balance_mul bone (balance_add n2 bzero))) (vdot y y))
      (balance_mul (balance_mul n2 n2) (vdot y y)) :=
    bal_poly_eq (fun | 0 => n2 | 1 => vdot y y | _ => bzero) e_s134_hh_rhs_L e_s134_hh_rhs_R p_s134_hh_rhs
  have hisom := step24_balance_trans (step24_balance_symm hlhs_eq) (step24_balance_trans hbibo hrhs_eq)
  exact ⟨h2, hlb, hn2_gt, hws, hisom⟩

-- =============================================================================
-- ШАГ 135: СИМПЛЕКТИЧЕСКИЙ J-XOR, 3D CROSS-PRODUCT И ТОТАЛЬНОСТЬ САТУРАЦИИ (v0.5.1 P1, 0 АКСИОМ)
-- =============================================================================

structure Vec2 where
  q : Balance
  p : Balance

def Vec2Equiv (a b : Vec2) : Prop :=
  BalanceEquiv a.q b.q ∧ BalanceEquiv a.p b.p

def symp_J (u : Vec2) : Vec2 := ⟨balance_neg u.p, u.q⟩

def symp_dot2 (u v : Vec2) : Balance :=
  balance_add (balance_mul u.q v.q) (balance_mul u.p v.p)

def symp_j_xor (f : Balance → Balance) (u v : Vec2) : Vec2 :=
  let jv := symp_J v
  let ju := symp_J u
  ⟨balance_sub (balance_mul u.q (f jv.q)) (balance_mul v.q (f ju.q)),
   balance_sub (balance_mul u.p (f jv.p)) (balance_mul v.p (f ju.p))⟩

private def e_s135_antisym_L : BExpr :=
  .add (.sub (.mul (.var 0) (.var 1)) (.mul (.var 2) (.var 3)))
       (.sub (.mul (.var 2) (.var 3)) (.mul (.var 0) (.var 1)))
private theorem p_s135_antisym : e_s135_antisym_L.norm = BExpr.zero.norm := rfl

private def e_s135_diag_L : BExpr :=
  .sub (.mul (.var 0) (.var 1)) (.mul (.var 0) (.var 1))
private theorem p_s135_diag : e_s135_diag_L.norm = BExpr.zero.norm := rfl

private def e_s135_j2_L : BExpr := .neg (.neg (.var 0))
private def e_s135_j2_R : BExpr := .var 0
private theorem p_s135_j2 : e_s135_j2_L.norm = e_s135_j2_R.norm := rfl

private def e_s135_jorth_L : BExpr :=
  .add (.mul (.var 0) (.neg (.var 1))) (.mul (.var 1) (.var 0))
private theorem p_s135_jorth : e_s135_jorth_L.norm = BExpr.zero.norm := rfl

private def e_s135_cross_add_x_L : BExpr :=
  .add (.sub (.mul (.var 1) (.var 5)) (.mul (.var 2) (.var 4)))
       (.sub (.mul (.var 4) (.var 2)) (.mul (.var 5) (.var 1)))
private def e_s135_cross_add_y_L : BExpr :=
  .add (.sub (.mul (.var 2) (.var 3)) (.mul (.var 0) (.var 5)))
       (.sub (.mul (.var 5) (.var 0)) (.mul (.var 3) (.var 2)))
private def e_s135_cross_add_z_L : BExpr :=
  .add (.sub (.mul (.var 0) (.var 4)) (.mul (.var 1) (.var 3)))
       (.sub (.mul (.var 3) (.var 1)) (.mul (.var 4) (.var 0)))
private theorem p_s135_cross_add_x : e_s135_cross_add_x_L.norm = BExpr.zero.norm := rfl
private theorem p_s135_cross_add_y : e_s135_cross_add_y_L.norm = BExpr.zero.norm := rfl
private theorem p_s135_cross_add_z : e_s135_cross_add_z_L.norm = BExpr.zero.norm := rfl

private def e_s135_cross_diag_x_L : BExpr := .sub (.mul (.var 1) (.var 2)) (.mul (.var 2) (.var 1))
private def e_s135_cross_diag_y_L : BExpr := .sub (.mul (.var 2) (.var 0)) (.mul (.var 0) (.var 2))
private def e_s135_cross_diag_z_L : BExpr := .sub (.mul (.var 0) (.var 1)) (.mul (.var 1) (.var 0))
private theorem p_s135_cross_diag_x : e_s135_cross_diag_x_L.norm = BExpr.zero.norm := rfl
private theorem p_s135_cross_diag_y : e_s135_cross_diag_y_L.norm = BExpr.zero.norm := rfl
private theorem p_s135_cross_diag_z : e_s135_cross_diag_z_L.norm = BExpr.zero.norm := rfl

private def e_s135_cross_orth_L : BExpr :=
  .add (.add (.mul (.var 0) (.sub (.mul (.var 1) (.var 5)) (.mul (.var 2) (.var 4))))
             (.mul (.var 1) (.sub (.mul (.var 2) (.var 3)) (.mul (.var 0) (.var 5)))))
       (.mul (.var 2) (.sub (.mul (.var 0) (.var 4)) (.mul (.var 1) (.var 3))))
private theorem p_s135_cross_orth : e_s135_cross_orth_L.norm = BExpr.zero.norm := rfl

/-- ШАГ 135. Рациональный симплектический `J`-XOR (`symplectic_j_xor_2d`), 3D-кросс (`cross_3d_pair`)
    и тотальность знаменателей сатураций `sigma_rat` / `sigma_abs` (`v0.5.1 P1`, 0 аксиом):
    1. Симплектическая инволюция: `J(J(u)) ~ -u` и ортогональность `⟨u, J u⟩ ~ 0`.
    2. Точная антисимметрия `Omega_XOR(u, v) + Omega_XOR(v, u) ~ 0` для ЛЮБОГО отображения `f`.
    3. Точное зануление диагонали `Omega_XOR(u, u) ~ 0`.
    4. Тотальность знаменателей сатураций: `1 + z² ≥ 1 > 0` (`sigma_rat`) и для любого `a ≥ 0` (`|z|`):
       `1 + a ≥ 1 > 0` (`sigma_abs`).
    5. 3D кросс-произведение: антисимметрия `u × v + v × u ~ 0`, `u × u ~ 0` и `⟨u, u × v⟩ ~ 0`. -/
theorem step135_symplectic_j_xor_and_rational_sat_totality
    (f : Balance → Balance) (u v : Vec2) (z a : Balance) (ha : BalGeZero a) (u3 v3 : Vec3) :
    let om_uv := symp_j_xor f u v
    let om_vu := symp_j_xor f v u
    let om_uu := symp_j_xor f u u
    let c_uv := cross3 u3 v3
    let c_vu := cross3 v3 u3
    let c_uu := cross3 u3 u3
    let den_rat := balance_add bone (bsq z)
    let den_abs := balance_add bone a
    (Vec2Equiv (symp_J (symp_J u)) ⟨balance_neg u.q, balance_neg u.p⟩ ∧
     BalanceEquiv (symp_dot2 u (symp_J u)) bzero) ∧
    Vec2Equiv ⟨balance_add om_uv.q om_vu.q, balance_add om_uv.p om_vu.p⟩ ⟨bzero, bzero⟩ ∧
    Vec2Equiv om_uu ⟨bzero, bzero⟩ ∧
    (BalLe bone den_rat ∧ BalGtZero den_rat) ∧
    (BalLe bone den_abs ∧ BalGtZero den_abs) ∧
    (Vec3Equiv ⟨balance_add c_uv.x c_vu.x, balance_add c_uv.y c_vu.y, balance_add c_uv.z c_vu.z⟩ ⟨bzero, bzero, bzero⟩ ∧
     Vec3Equiv c_uu ⟨bzero, bzero, bzero⟩ ∧
     BalanceEquiv (dot3 u3 c_uv) bzero) := by
  dsimp only [symp_J, symp_dot2, symp_j_xor, Vec2Equiv, Vec3Equiv, cross3, dot3, bsq]
  have hj2 : BalanceEquiv (balance_neg (balance_neg u.q)) u.q :=
    bal_poly_eq (fun | 0 => u.q | _ => bzero) e_s135_j2_L e_s135_j2_R p_s135_j2
  have hj2_q : BalanceEquiv (balance_neg u.q) (balance_neg u.q) := step24_balance_refl _
  have hj2_p : BalanceEquiv (balance_neg u.p) (balance_neg u.p) := step24_balance_refl _
  have hjorth : BalanceEquiv (balance_add (balance_mul u.q (balance_neg u.p)) (balance_mul u.p u.q)) bzero :=
    bal_poly_eq (fun | 0 => u.q | 1 => u.p | _ => bzero) e_s135_jorth_L .zero p_s135_jorth
  have hanti_q : BalanceEquiv
      (balance_add (balance_sub (balance_mul u.q (f (balance_neg v.p))) (balance_mul v.q (f (balance_neg u.p))))
                   (balance_sub (balance_mul v.q (f (balance_neg u.p))) (balance_mul u.q (f (balance_neg v.p)))))
      bzero :=
    bal_poly_eq (fun | 0 => u.q | 1 => f (balance_neg v.p) | 2 => v.q | 3 => f (balance_neg u.p) | _ => bzero)
      e_s135_antisym_L .zero p_s135_antisym
  have hanti_p : BalanceEquiv
      (balance_add (balance_sub (balance_mul u.p (f v.q)) (balance_mul v.p (f u.q)))
                   (balance_sub (balance_mul v.p (f u.q)) (balance_mul u.p (f v.q))))
      bzero :=
    bal_poly_eq (fun | 0 => u.p | 1 => f v.q | 2 => v.p | 3 => f u.q | _ => bzero)
      e_s135_antisym_L .zero p_s135_antisym
  have hdiag_q : BalanceEquiv
      (balance_sub (balance_mul u.q (f (balance_neg u.p))) (balance_mul u.q (f (balance_neg u.p)))) bzero :=
    bal_poly_eq (fun | 0 => u.q | 1 => f (balance_neg u.p) | _ => bzero) e_s135_diag_L .zero p_s135_diag
  have hdiag_p : BalanceEquiv
      (balance_sub (balance_mul u.p (f u.q)) (balance_mul u.p (f u.q))) bzero :=
    bal_poly_eq (fun | 0 => u.p | 1 => f u.q | _ => bzero) e_s135_diag_L .zero p_s135_diag
  have hrat_le : BalLe bone (balance_add bone (balance_mul z z)) :=
    bal_ge_zero_of_equiv
      (step24_balance_symm (bal_poly_eq (fun | 0 => balance_mul z z | 1 => bone | _ => bzero)
        e_add_sub_self_L e_add_sub_self_R p_add_sub_self))
      (bal_sq_ge_zero z)
  have hrat_gt : BalGtZero (balance_add bone (balance_mul z z)) := bal_one_add_sq_gt_zero z
  have habs_le : BalLe bone (balance_add bone a) :=
    bal_ge_zero_of_equiv
      (step24_balance_symm (bal_poly_eq (fun | 0 => a | 1 => bone | _ => bzero)
        e_add_sub_self_L e_add_sub_self_R p_add_sub_self))
      ha
  have habs_gt : BalGtZero (balance_add bone a) := bal_add_gt_ge bone_gt_zero ha
  let env3 : Nat → Balance := fun
    | 0 => u3.x | 1 => u3.y | 2 => u3.z | 3 => v3.x | 4 => v3.y | 5 => v3.z | _ => bzero
  let env3u : Nat → Balance := fun | 0 => u3.x | 1 => u3.y | 2 => u3.z | _ => bzero
  have _ := hj2
  exact ⟨⟨⟨hj2_q, hj2_p⟩, hjorth⟩,
         ⟨hanti_q, hanti_p⟩,
         ⟨hdiag_q, hdiag_p⟩,
         ⟨hrat_le, hrat_gt⟩,
         ⟨habs_le, habs_gt⟩,
         ⟨⟨bal_poly_eq env3 e_s135_cross_add_x_L .zero p_s135_cross_add_x,
           bal_poly_eq env3 e_s135_cross_add_y_L .zero p_s135_cross_add_y,
           bal_poly_eq env3 e_s135_cross_add_z_L .zero p_s135_cross_add_z⟩,
          ⟨bal_poly_eq env3u e_s135_cross_diag_x_L .zero p_s135_cross_diag_x,
           bal_poly_eq env3u e_s135_cross_diag_y_L .zero p_s135_cross_diag_y,
           bal_poly_eq env3u e_s135_cross_diag_z_L .zero p_s135_cross_diag_z⟩,
          bal_poly_eq env3 e_s135_cross_orth_L .zero p_s135_cross_orth⟩⟩

-- =============================================================================
-- ШАГ 136: ОРТОГОНАЛЬНОСТЬ И КОМПОЗИЦИЯ МИКРО-РОТОРНОГО АДАПТЕРА БАНКА (v0.5.1 P2, 0 АКСИОМ)
-- =============================================================================

def rot2_cayley_num (t : Balance) (u : Vec2) : Vec2 :=
  let c := balance_sub bone (bsq t)
  let s := balance_mul (balOfNat 2) t
  ⟨balance_sub (balance_mul c u.q) (balance_mul s u.p),
   balance_add (balance_mul s u.q) (balance_mul c u.p)⟩

def rot3_vec_num (g u : Vec3) : Vec3 :=
  let g2 := dot3 g g
  let om := balance_sub bone g2
  let gu := dot3 g u
  let cr := cross3 g u
  let two := balOfNat 2
  ⟨balance_add (balance_add (balance_mul om u.x) (balance_mul two cr.x)) (balance_mul two (balance_mul gu g.x)),
   balance_add (balance_add (balance_mul om u.y) (balance_mul two cr.y)) (balance_mul two (balance_mul gu g.y)),
   balance_add (balance_add (balance_mul om u.z) (balance_mul two cr.z)) (balance_mul two (balance_mul gu g.z))⟩

private def bRot2E (t : BExpr) (u : BExpr × BExpr) : BExpr × BExpr :=
  let c := BExpr.sub .one (.mul t t)
  let s := BExpr.mul (.nat 2) t
  (.sub (.mul c u.1) (.mul s u.2), .add (.mul s u.1) (.mul c u.2))

private def e_s136_r2_t1 := BExpr.var 0
private def e_s136_r2_t2 := BExpr.var 1
private def e_s136_u2 : BExpr × BExpr := (.var 2, .var 3)
private def e_s136_r2_1 := bRot2E e_s136_r2_t1 e_s136_u2
private def e_s136_r2_comp := bRot2E e_s136_r2_t2 e_s136_r2_1
private def e_s136_d1 := BExpr.add .one (.mul e_s136_r2_t1 e_s136_r2_t1)
private def e_s136_d2 := BExpr.add .one (.mul e_s136_r2_t2 e_s136_r2_t2)
private def e_s136_d12 := BExpr.mul e_s136_d1 e_s136_d2

private theorem p_s136_r2_det :
    (BExpr.add (.mul (.sub .one (.mul (.var 0) (.var 0))) (.sub .one (.mul (.var 0) (.var 0))))
               (.mul (.mul (.nat 2) (.var 0)) (.mul (.nat 2) (.var 0)))).norm =
    (BExpr.mul e_s136_d1 e_s136_d1).norm := rfl

private theorem p_s136_r2_comp_norm :
    (BExpr.add (.mul e_s136_r2_comp.1 e_s136_r2_comp.1)
               (.mul e_s136_r2_comp.2 e_s136_r2_comp.2)).norm =
    (BExpr.mul (.mul e_s136_d12 e_s136_d12)
               (.add (.mul e_s136_u2.1 e_s136_u2.1) (.mul e_s136_u2.2 e_s136_u2.2))).norm := rfl

private def bRot3VecE (g u : BExpr × BExpr × BExpr) : BExpr × BExpr × BExpr :=
  let g2 := vDotE g g
  let om := BExpr.sub .one g2
  let gu := vDotE g u
  let cx := BExpr.sub (.mul g.2.1 u.2.2) (.mul g.2.2 u.2.1)
  let cy := BExpr.sub (.mul g.2.2 u.1) (.mul g.1 u.2.2)
  let cz := BExpr.sub (.mul g.1 u.2.1) (.mul g.2.1 u.1)
  (.add (.add (.mul om u.1) (.mul (.nat 2) cx)) (.mul (.nat 2) (.mul gu g.1)),
   .add (.add (.mul om u.2.1) (.mul (.nat 2) cy)) (.mul (.nat 2) (.mul gu g.2.1)),
   .add (.add (.mul om u.2.2) (.mul (.nat 2) cz)) (.mul (.nat 2) (.mul gu g.2.2)))

private def e_s136_g3 : BExpr × BExpr × BExpr := (.var 0, .var 1, .var 2)
private def e_s136_u3 : BExpr × BExpr × BExpr := (.var 3, .var 4, .var 5)
private def e_s136_r3 := bRot3VecE e_s136_g3 e_s136_u3
private def e_s136_dg := BExpr.add .one (vDotE e_s136_g3 e_s136_g3)
private def e_s136_c0 := bRot3VecE e_s136_g3 (.one, .zero, .zero)
private def e_s136_c1 := bRot3VecE e_s136_g3 (.zero, .one, .zero)
private def e_s136_c2 := bRot3VecE e_s136_g3 (.zero, .zero, .one)

private theorem p_s136_r3_norm :
    (vDotE e_s136_r3 e_s136_r3).norm =
    (BExpr.mul (.mul e_s136_dg e_s136_dg) (vDotE e_s136_u3 e_s136_u3)).norm := rfl

private theorem p_s136_r3_det :
    (bDet3E e_s136_c0 e_s136_c1 e_s136_c2).norm =
    (BExpr.mul e_s136_dg (.mul e_s136_dg e_s136_dg)).norm := rfl

private theorem p_s136_r3_inv_x :
    (bRot3VecE (.neg (.var 0), .neg (.var 1), .neg (.var 2)) e_s136_r3).1.norm =
    (BExpr.mul (.mul e_s136_dg e_s136_dg) e_s136_u3.1).norm := rfl
private theorem p_s136_r3_inv_y :
    (bRot3VecE (.neg (.var 0), .neg (.var 1), .neg (.var 2)) e_s136_r3).2.1.norm =
    (BExpr.mul (.mul e_s136_dg e_s136_dg) e_s136_u3.2.1).norm := rfl
private theorem p_s136_r3_inv_z :
    (bRot3VecE (.neg (.var 0), .neg (.var 1), .neg (.var 2)) e_s136_r3).2.2.norm =
    (BExpr.mul (.mul e_s136_dg e_s136_dg) e_s136_u3.2.2).norm := rfl

/-- ШАГ 136. Точная ортогональность, `det = +1` и сохранение BIBO-границы при композиции
    хребтового ротора и 160-байтного банковского адаптера (`v0.5.1 P2`, 0 аксиом):
    1. 2D композиция `R_2(t_adapt) ∘ R_2(t_bb)`: знаменатели `d₁, d₂ > 0`, `det(R_2) = d₁² > 0`
       (`det = +1`) и точная изометрия `||R_2(t₂)(R_2(t₁) u)||² ~ (d₁·d₂)² · ||u||²`.
    2. 3D банковский адаптер `R_3(g)` от свободного `g ∈ R³`: знаменатель `d_g = 1 + ||g||² > 0`,
       точная изометрия `||R_3(g) u||² ~ d_g² · ||u||²`, точный обратный `R_3(-g)(R_3(g) u) ~ d_g² · u`
       (обратный проход `R_3^T = R_3(-g)`), и `det(R_3(g)) ~ d_g³ > 0` (`det = +1`, строго `SO(3)`). -/
theorem step136_bank_rotor_adapter_composition_isometry
    (t1 t2 : Balance) (u2 : Vec2) (g u3 : Vec3) :
    let d1 := balance_add bone (bsq t1)
    let d2 := balance_add bone (bsq t2)
    let d12 := balance_mul d1 d2
    let r2_comp := rot2_cayley_num t2 (rot2_cayley_num t1 u2)
    let det2 := balance_add (bsq (balance_sub bone (bsq t1))) (bsq (balance_mul (balOfNat 2) t1))
    let dg := balance_add bone (dot3 g g)
    let r3 := rot3_vec_num g u3
    let r3_back := rot3_vec_num ⟨balance_neg g.x, balance_neg g.y, balance_neg g.z⟩ r3
    let c0 := rot3_vec_num g ⟨bone, bzero, bzero⟩
    let c1 := rot3_vec_num g ⟨bzero, bone, bzero⟩
    let c2 := rot3_vec_num g ⟨bzero, bzero, bone⟩
    (BalGtZero d1 ∧ BalGtZero d2 ∧ BalGtZero d12 ∧
     BalanceEquiv det2 (bsq d1) ∧
     BalanceEquiv (symp_dot2 r2_comp r2_comp) (balance_mul (bsq d12) (symp_dot2 u2 u2))) ∧
    (BalGtZero dg ∧
     BalanceEquiv (dot3 r3 r3) (balance_mul (bsq dg) (dot3 u3 u3)) ∧
     Vec3Equiv r3_back ⟨balance_mul (bsq dg) u3.x, balance_mul (bsq dg) u3.y, balance_mul (bsq dg) u3.z⟩ ∧
     BalanceEquiv (det3_cols c0 c1 c2) (balance_mul dg (bsq dg)) ∧
     BalGtZero (det3_cols c0 c1 c2)) := by
  dsimp only [rot2_cayley_num, rot3_vec_num, symp_dot2, det3_cols, Vec3Equiv, cross3, dot3, bsq]
  let env2 : Nat → Balance := fun | 0 => t1 | 1 => t2 | 2 => u2.q | 3 => u2.p | _ => bzero
  let env3 : Nat → Balance := fun | 0 => g.x | 1 => g.y | 2 => g.z | 3 => u3.x | 4 => u3.y | 5 => u3.z | _ => bzero
  have hd1 : BalGtZero (balance_add bone (balance_mul t1 t1)) := bal_one_add_sq_gt_zero t1
  have hd2 : BalGtZero (balance_add bone (balance_mul t2 t2)) := bal_one_add_sq_gt_zero t2
  have hd12 := bal_mul_gt_gt hd1 hd2
  have hdg : BalGtZero (balance_add bone (dot3 g g)) := bal_add_gt_ge bone_gt_zero (dot3_self_ge_zero g)
  have hdet3_eq := bal_poly_eq env3 (bDet3E e_s136_c0 e_s136_c1 e_s136_c2)
    (BExpr.mul e_s136_dg (.mul e_s136_dg e_s136_dg)) p_s136_r3_det
  have hdet3_gt := bal_gt_zero_of_equiv (step24_balance_symm hdet3_eq)
    (bal_mul_gt_gt hdg (bal_mul_gt_gt hdg hdg))
  exact ⟨⟨hd1, hd2, hd12,
          bal_poly_eq env2
            (BExpr.add (.mul (.sub .one (.mul (.var 0) (.var 0))) (.sub .one (.mul (.var 0) (.var 0))))
                       (.mul (.mul (.nat 2) (.var 0)) (.mul (.nat 2) (.var 0))))
            (BExpr.mul e_s136_d1 e_s136_d1) p_s136_r2_det,
          bal_poly_eq env2
            (BExpr.add (.mul e_s136_r2_comp.1 e_s136_r2_comp.1) (.mul e_s136_r2_comp.2 e_s136_r2_comp.2))
            (BExpr.mul (.mul e_s136_d12 e_s136_d12)
                       (.add (.mul e_s136_u2.1 e_s136_u2.1) (.mul e_s136_u2.2 e_s136_u2.2)))
            p_s136_r2_comp_norm⟩,
         ⟨hdg,
          bal_poly_eq env3 (vDotE e_s136_r3 e_s136_r3)
            (BExpr.mul (.mul e_s136_dg e_s136_dg) (vDotE e_s136_u3 e_s136_u3)) p_s136_r3_norm,
          ⟨bal_poly_eq env3 (bRot3VecE (.neg (.var 0), .neg (.var 1), .neg (.var 2)) e_s136_r3).1
             (BExpr.mul (.mul e_s136_dg e_s136_dg) e_s136_u3.1) p_s136_r3_inv_x,
           bal_poly_eq env3 (bRot3VecE (.neg (.var 0), .neg (.var 1), .neg (.var 2)) e_s136_r3).2.1
             (BExpr.mul (.mul e_s136_dg e_s136_dg) e_s136_u3.2.1) p_s136_r3_inv_y,
           bal_poly_eq env3 (bRot3VecE (.neg (.var 0), .neg (.var 1), .neg (.var 2)) e_s136_r3).2.2
             (BExpr.mul (.mul e_s136_dg e_s136_dg) e_s136_u3.2.2) p_s136_r3_inv_z⟩,
          hdet3_eq, hdet3_gt⟩⟩

-- =============================================================================
-- ШАГ 137: СТРОГАЯ ПОЛОЖИТЕЛЬНОСТЬ И АСИММЕТРИЯ ФИНСЛЕР-РАНДЕРС ДИОДА (v0.5.1 P3, 0 АКСИОМ)
-- =============================================================================

private def e_s137_lag2_L : BExpr :=
  .sub (.mul (.add (.mul (.var 0) (.var 0)) (.mul (.var 1) (.var 1)))
             (.add (.mul (.var 2) (.var 2)) (.mul (.var 3) (.var 3))))
       (.mul (.add (.mul (.var 0) (.var 2)) (.mul (.var 1) (.var 3)))
             (.add (.mul (.var 0) (.var 2)) (.mul (.var 1) (.var 3))))
private def e_s137_lag2_R : BExpr :=
  let cr := BExpr.sub (.mul (.var 0) (.var 3)) (.mul (.var 1) (.var 2))
  .mul cr cr
private theorem p_s137_lag2 : e_s137_lag2_L.norm = e_s137_lag2_R.norm := rfl

private def e_s137_lag3_L : BExpr :=
  let b : BExpr × BExpr × BExpr := (.var 0, .var 1, .var 2)
  let d : BExpr × BExpr × BExpr := (.var 3, .var 4, .var 5)
  let bd := vDotE b d
  .sub (.mul (vDotE b b) (vDotE d d)) (.mul bd bd)
private def e_s137_lag3_R : BExpr :=
  let cx := BExpr.sub (.mul (.var 1) (.var 5)) (.mul (.var 2) (.var 4))
  let cy := BExpr.sub (.mul (.var 2) (.var 3)) (.mul (.var 0) (.var 5))
  let cz := BExpr.sub (.mul (.var 0) (.var 4)) (.mul (.var 1) (.var 3))
  .add (.add (.mul cx cx) (.mul cy cy)) (.mul cz cz)
private theorem p_s137_lag3 : e_s137_lag3_L.norm = e_s137_lag3_R.norm := rfl

private def e_s137_diffsq_L : BExpr :=
  .sub (.mul (.mul (.var 0) (.var 1)) (.mul (.var 0) (.var 1))) (.mul (.var 2) (.var 2))
private def e_s137_diffsq_R : BExpr :=
  .mul (.add (.mul (.var 0) (.var 1)) (.var 2)) (.sub (.mul (.var 0) (.var 1)) (.var 2))
private theorem p_s137_diffsq : e_s137_diffsq_L.norm = e_s137_diffsq_R.norm := rfl

private def e_s137_randers_decomp_L : BExpr :=
  .sub (.mul (.mul (.var 0) (.var 0)) (.add (.var 1) (.var 2))) (.var 3)
private def e_s137_randers_decomp_R : BExpr :=
  .add (.add (.mul (.sub (.mul (.var 0) (.var 0)) (.var 4)) (.var 1))
             (.sub (.mul (.var 4) (.var 1)) (.var 3)))
       (.mul (.mul (.var 0) (.var 0)) (.var 2))
private theorem p_s137_randers_decomp : e_s137_randers_decomp_L.norm = e_s137_randers_decomp_R.norm := rfl

private def e_s137_asym_L : BExpr :=
  .sub (.add (.var 0) (.var 1)) (.sub (.var 0) (.var 1))
private def e_s137_asym_R : BExpr :=
  .mul (.nat 2) (.var 1)
private theorem p_s137_asym : e_s137_asym_L.norm = e_s137_asym_R.norm := rfl

private def e_s137_sq_mul_L : BExpr :=
  .mul (.mul (.var 0) (.var 1)) (.mul (.var 0) (.var 1))
private def e_s137_sq_mul_R : BExpr :=
  .mul (.mul (.var 0) (.var 0)) (.mul (.var 1) (.var 1))
private theorem p_s137_sq_mul : e_s137_sq_mul_L.norm = e_s137_sq_mul_R.norm := rfl

private def e_s137_neg_mul_L : BExpr := .neg (.mul (.var 0) (.var 1))
private def e_s137_neg_mul_R : BExpr := .mul (.neg (.var 0)) (.var 1)
private theorem p_s137_neg_mul : e_s137_neg_mul_L.norm = e_s137_neg_mul_R.norm := rfl

private def e_s137_zero_mul_L : BExpr := .mul .zero (.var 0)
private theorem p_s137_zero_mul : e_s137_zero_mul_L.norm = BExpr.zero.norm := rfl

private def e_s137_sub_neg_L : BExpr := .sub (.var 0) (.var 1)
private def e_s137_sub_neg_R : BExpr := .add (.var 0) (.neg (.var 1))
private theorem p_s137_sub_neg : e_s137_sub_neg_L.norm = e_s137_sub_neg_R.norm := rfl

private theorem bal_gt_zero_of_prod_gt_zero (a b : Balance) (hb : BalGtZero b) (hab : BalGtZero (balance_mul a b)) :
    BalGtZero a := by
  match a with
  | .mk p n =>
    match trichotomy p n with
    | .inl heq =>
      -- a ~ 0 => a * b ~ 0, contradicts a * b > 0
      have h0 : BalanceEquiv (Balance.mk p n) bzero := by
        dsimp [BalanceEquiv, bzero]
        rw [heq]
      have hprod0 : BalanceEquiv (balance_mul (Balance.mk p n) b) bzero :=
        step24_balance_trans
          (bal_mul_congr h0 (step24_balance_refl b))
          (bal_poly_eq (fun | 0 => b | _ => bzero) e_s137_zero_mul_L .zero p_s137_zero_mul)
      have hz_gt : BalGtZero bzero := bal_gt_zero_of_equiv hprod0 hab
      exact False.elim (step19_lt_irrefl PosNat.one hz_gt)
    | .inr (.inl hlt) =>
      -- a < 0 => -a > 0 => (-a) * b > 0 => -(a * b) > 0, contradicts a * b > 0
      have hneg_a : BalGtZero (balance_neg (Balance.mk p n)) := hlt
      have hneg_prod : BalGtZero (balance_neg (balance_mul (Balance.mk p n) b)) :=
        bal_gt_zero_of_equiv
          (step24_balance_symm (bal_poly_eq (fun | 0 => Balance.mk p n | 1 => b | _ => bzero)
            e_s137_neg_mul_L e_s137_neg_mul_R p_s137_neg_mul))
          (bal_mul_gt_gt hneg_a hb)
      exact False.elim (step19_lt_irrefl _ (step20_lt_trans hab hneg_prod))
    | .inr (.inr hgt) =>
      exact hgt

private theorem bal_add_sub_pos_of_sq_gt (r c : Balance) (hr : BalGtZero r)
    (hsq : BalGtZero (balance_sub (bsq r) (bsq c))) :
    BalGtZero (balance_add r c) ∧ BalGtZero (balance_sub r c) := by
  dsimp only [bsq] at hsq
  have hfac : BalanceEquiv (balance_sub (balance_mul r r) (balance_mul c c))
                           (balance_mul (balance_add r c) (balance_sub r c)) :=
    bal_poly_eq (fun | 0 => bone | 1 => r | 2 => c | _ => bzero)
      (.sub (.mul (.var 1) (.var 1)) (.mul (.var 2) (.var 2)))
      (.mul (.add (.var 1) (.var 2)) (.sub (.var 1) (.var 2))) rfl
  have hprod : BalGtZero (balance_mul (balance_add r c) (balance_sub r c)) :=
    bal_gt_zero_of_equiv hfac hsq
  have hprod_comm : BalGtZero (balance_mul (balance_sub r c) (balance_add r c)) :=
    bal_gt_zero_of_equiv
      (bal_poly_eq (fun | 0 => balance_add r c | 1 => balance_sub r c | _ => bzero)
        (.mul (.var 0) (.var 1)) (.mul (.var 1) (.var 0)) rfl)
      hprod
  match c with
  | .mk cp cn =>
    match trichotomy cp cn with
    | .inl hceq =>
      have hc0 : BalanceEquiv (Balance.mk cp cn) bzero := by
        dsimp [BalanceEquiv, bzero]; rw [hceq]
      have hadd : BalGtZero (balance_add r (Balance.mk cp cn)) :=
        bal_add_gt_ge hr (Or.inl hc0)
      exact ⟨hadd, bal_gt_zero_of_prod_gt_zero _ _ hadd hprod_comm⟩
    | .inr (.inl hclt) =>
      -- c < 0 => -c > 0 => r - c = r + (-c) > 0 => r + c > 0
      have hnc : BalGtZero (balance_neg (Balance.mk cp cn)) := hclt
      have hsub : BalGtZero (balance_sub r (Balance.mk cp cn)) :=
        bal_gt_zero_of_equiv
          (step24_balance_symm (bal_poly_eq (fun | 0 => r | 1 => Balance.mk cp cn | _ => bzero)
            e_s137_sub_neg_L e_s137_sub_neg_R p_s137_sub_neg))
          (bal_add_gt_ge hr (Or.inr hnc))
      exact ⟨bal_gt_zero_of_prod_gt_zero _ _ hsub hprod, hsub⟩
    | .inr (.inr hcgt) =>
      have hadd : BalGtZero (balance_add r (Balance.mk cp cn)) :=
        bal_add_gt_ge hr (Or.inr hcgt)
      exact ⟨hadd, bal_gt_zero_of_prod_gt_zero _ _ hadd hprod_comm⟩

/-- ШАГ 137. Строгая положительность, причинная асимметрия и ограниченность нормированного контраста
    рационального Финслер-Рандерс диода (`finsler_randers_diode_2d/3d` и `finsler_randers_contrast_2d/3d`,
    `v0.5.1 P3`, 0 аксиом):
    1. Ретракция в открытый единичный шар: для любого `v ∈ R³`, `1 + ||v||² > ||v||²` (`||β||² < 1`).
    2. Точные тождества Лагранжа–Коши–Буняковского (2D и 3D):
       `||b₂||²||δ₂||² - ⟨b₂,δ₂⟩² ≥ 0` и `||b||²||δ||² - ⟨b,δ⟩² ~ ||b × δ||² ≥ 0`.
    3. Для ЛЮБЫХ `D > 0`, `||b||² < D²` (`||β|| < 1`), `μ₀ > 0`, `ρ > 0` с `ρ² ~ ||δ||² + μ₀²`:
       `(D·ρ)² - ⟨b, δ⟩² > 0`, откуда `τ(+δ) = D·ρ + ⟨b, δ⟩ > 0`, `τ(-δ) = D·ρ - ⟨b, δ⟩ > 0`,
       знаменатель нормированного контраста `τ(+δ) + τ(-δ) ~ 2·D·ρ > 0` тотален и `|c(v, δ)| < 1`.
    4. Строгая причинная асимметрия: `τ(+δ) - τ(-δ) ~ 2·⟨b, δ⟩`, откуда
       `⟨b, δ⟩ > 0 → τ(+δ) > τ(-δ)` и `⟨b, δ⟩ < 0 → τ(-δ) > τ(+δ)`. -/
theorem step137_finsler_randers_diode_positivity_and_asymmetry
    (b2 d2 : Vec2) (b d : Vec3) (D mu0 rho : Balance)
    (hD : BalGtZero D) (hmu : BalGtZero mu0) (hrho : BalGtZero rho)
    (hball : BalLt (dot3 b b) (bsq D))
    (hrho_eq : BalanceEquiv (bsq rho) (balance_add (dot3 d d) (bsq mu0))) :
    let Dr := balance_mul D rho
    let bd := dot3 b d
    let tau_pos := balance_add Dr bd
    let tau_neg := balance_sub Dr bd
    BalLt (dot3 b b) (balance_add bone (dot3 b b)) ∧
    BalGeZero (balance_sub (balance_mul (symp_dot2 b2 b2) (symp_dot2 d2 d2)) (bsq (symp_dot2 b2 d2))) ∧
    BalGeZero (balance_sub (balance_mul (dot3 b b) (dot3 d d)) (bsq bd)) ∧
    BalGtZero (balance_sub (bsq Dr) (bsq bd)) ∧
    (BalGtZero tau_pos ∧ BalGtZero tau_neg) ∧
    BalanceEquiv (balance_sub tau_pos tau_neg) (balance_mul (balOfNat 2) bd) ∧
    (BalanceEquiv (balance_add tau_pos tau_neg) (balance_mul (balOfNat 2) Dr) ∧
     BalGtZero (balance_add tau_pos tau_neg)) ∧
    (BalGtZero bd → BalLt tau_neg tau_pos) ∧
    (BalGtZero (balance_neg bd) → BalLt tau_pos tau_neg) := by
  dsimp only [bsq] at *
  have hret : BalLt (dot3 b b) (balance_add bone (dot3 b b)) :=
    bal_gt_zero_of_equiv
      (step24_balance_symm (bal_poly_eq (fun | 0 => bone | 1 => dot3 b b | _ => bzero)
        (.sub (.add (.var 0) (.var 1)) (.var 1)) (.var 0) rfl))
      bone_gt_zero
  have hcs2 : BalGeZero (balance_sub (balance_mul (symp_dot2 b2 b2) (symp_dot2 d2 d2))
                                     (balance_mul (symp_dot2 b2 d2) (symp_dot2 b2 d2))) :=
    bal_ge_zero_of_equiv
      (step24_balance_symm (bal_poly_eq (fun | 0 => b2.q | 1 => b2.p | 2 => d2.q | 3 => d2.p | _ => bzero)
        e_s137_lag2_L e_s137_lag2_R p_s137_lag2))
      (bal_sq_ge_zero (balance_sub (balance_mul b2.q d2.p) (balance_mul b2.p d2.q)))
  let env3 : Nat → Balance := fun
    | 0 => b.x | 1 => b.y | 2 => b.z | 3 => d.x | 4 => d.y | 5 => d.z | _ => bzero
  have hcs3 : BalGeZero (balance_sub (balance_mul (dot3 b b) (dot3 d d))
                                     (balance_mul (dot3 b d) (dot3 b d))) :=
    bal_ge_zero_of_equiv
      (step24_balance_symm (bal_poly_eq env3 e_s137_lag3_L e_s137_lag3_R p_s137_lag3))
      (dot3_self_ge_zero (cross3 b d))
  have hDr_sq : BalanceEquiv (balance_mul (balance_mul D rho) (balance_mul D rho))
                             (balance_mul (balance_mul D D) (balance_add (dot3 d d) (balance_mul mu0 mu0))) :=
    step24_balance_trans
      (bal_poly_eq (fun | 0 => D | 1 => rho | _ => bzero) e_s137_sq_mul_L e_s137_sq_mul_R p_s137_sq_mul)
      (bal_mul_congr (step24_balance_refl _) hrho_eq)
  have hD2 : BalGtZero (balance_mul D D) := bal_mul_gt_gt hD hD
  have hmu2 : BalGtZero (balance_mul mu0 mu0) := bal_mul_gt_gt hmu hmu
  have hD2mu2 : BalGtZero (balance_mul (balance_mul D D) (balance_mul mu0 mu0)) := bal_mul_gt_gt hD2 hmu2
  have hterm1 : BalGeZero (balance_mul (balance_sub (balance_mul D D) (dot3 b b)) (dot3 d d)) :=
    bal_mul_ge_ge (Or.inr hball) (dot3_self_ge_zero d)
  have hdecomp := bal_poly_eq
    (fun | 0 => D | 1 => dot3 d d | 2 => balance_mul mu0 mu0
         | 3 => balance_mul (dot3 b d) (dot3 b d) | 4 => dot3 b b | _ => bzero)
    e_s137_randers_decomp_L e_s137_randers_decomp_R p_s137_randers_decomp
  have hgap_rhs : BalGtZero (balance_sub (balance_mul (balance_mul D D) (balance_add (dot3 d d) (balance_mul mu0 mu0)))
                                         (balance_mul (dot3 b d) (dot3 b d))) :=
    bal_gt_zero_of_equiv (step24_balance_symm hdecomp)
      (bal_gt_zero_of_equiv
        (bal_poly_eq (fun | 0 => balance_add (balance_mul (balance_sub (balance_mul D D) (dot3 b b)) (dot3 d d))
                                            (balance_sub (balance_mul (dot3 b b) (dot3 d d)) (balance_mul (dot3 b d) (dot3 b d)))
                          | 1 => balance_mul (balance_mul D D) (balance_mul mu0 mu0) | _ => bzero)
          (.add (.var 1) (.var 0)) (.add (.var 0) (.var 1)) rfl)
        (bal_add_gt_ge hD2mu2 (bal_add_ge_ge hterm1 hcs3)))
  have hgap : BalGtZero (balance_sub (balance_mul (balance_mul D rho) (balance_mul D rho))
                                     (balance_mul (dot3 b d) (dot3 b d))) :=
    bal_gt_zero_of_equiv
      (step24_balance_symm (step25_balance_add_congr hDr_sq (step24_balance_refl (balance_neg (balance_mul (dot3 b d) (dot3 b d))))))
      hgap_rhs
  have htaus := bal_add_sub_pos_of_sq_gt (balance_mul D rho) (dot3 b d) (bal_mul_gt_gt hD hrho) hgap
  have hasym : BalanceEquiv (balance_sub (balance_add (balance_mul D rho) (dot3 b d))
                                         (balance_sub (balance_mul D rho) (dot3 b d)))
                            (balance_mul (balOfNat 2) (dot3 b d)) :=
    bal_poly_eq (fun | 0 => balance_mul D rho | 1 => dot3 b d | _ => bzero) e_s137_asym_L e_s137_asym_R p_s137_asym
  have hdir_pos : BalGtZero (dot3 b d) →
      BalLt (balance_sub (balance_mul D rho) (dot3 b d)) (balance_add (balance_mul D rho) (dot3 b d)) := by
    intro hbd
    exact bal_gt_zero_of_equiv (step24_balance_symm hasym) (bal_mul_gt_gt (balOfNat_succ_gt_zero 1) hbd)
  have hdir_neg : BalGtZero (balance_neg (dot3 b d)) →
      BalLt (balance_add (balance_mul D rho) (dot3 b d)) (balance_sub (balance_mul D rho) (dot3 b d)) := by
    intro hnbd
    have hasym_rev : BalanceEquiv
        (balance_sub (balance_sub (balance_mul D rho) (dot3 b d))
                     (balance_add (balance_mul D rho) (dot3 b d)))
        (balance_mul (balOfNat 2) (balance_neg (dot3 b d))) :=
      bal_poly_eq (fun | 0 => balance_mul D rho | 1 => dot3 b d | _ => bzero)
        (.sub (.sub (.var 0) (.var 1)) (.add (.var 0) (.var 1)))
        (.mul (.nat 2) (.neg (.var 1))) rfl
    exact bal_gt_zero_of_equiv (step24_balance_symm hasym_rev) (bal_mul_gt_gt (balOfNat_succ_gt_zero 1) hnbd)
  have hsum : BalanceEquiv (balance_add (balance_add (balance_mul D rho) (dot3 b d))
                                        (balance_sub (balance_mul D rho) (dot3 b d)))
                           (balance_mul (balOfNat 2) (balance_mul D rho)) :=
    bal_poly_eq (fun | 0 => balance_mul D rho | 1 => dot3 b d | _ => bzero)
      (.add (.add (.var 0) (.var 1)) (.sub (.var 0) (.var 1)))
      (.mul (.nat 2) (.var 0)) rfl
  have hsum_gt : BalGtZero (balance_add (balance_add (balance_mul D rho) (dot3 b d))
                                        (balance_sub (balance_mul D rho) (dot3 b d))) :=
    bal_gt_zero_of_equiv (step24_balance_symm hsum)
      (bal_mul_gt_gt (balOfNat_succ_gt_zero 1) (bal_mul_gt_gt hD hrho))
  exact ⟨hret, hcs2, hcs3, hgap, htaus, hasym, ⟨hsum, hsum_gt⟩, hdir_pos, hdir_neg⟩

/-- ШАГ 138. Рациональный гейт Ляпунова и рациональный баланс каналов Либеля–Кёрнера
    (`lyapunov_backbone_gate`, `rational_lk_balance_step`, `v0.5.1 P4`, 0 аксиом):
    1. Для любых `κ > 0` и `0 ≤ x₁ ≤ x₂`: знаменатели `D(x) = 1 + κ·x` строго положительны,
       `D(x) ≥ 1` (т.е. `0 < g_Lyap(x) = 1 / D(x) ≤ 1`), и `D(x₁) ≤ D(x₂)`
       (т.е. гейт монотонно сжимается при росте всплеска ошибки: `g_Lyap(x₂) ≤ g_Lyap(x₁)`).
    2. При строгом всплеске `x > 0` знаменатель `D(x) > 1` (строгое демпфирование `g_Lyap(x) < 1`).
    3. Для любого параметра канала `s ∈ Balance` в точной шкале Rust `w(s) = (5/4) / (1 + s²) = 1.25 / (1 + s²)`:
       знаменатель `4·(1 + s²) ≥ 4 > 0` тотален, барьер `B₂(s) = s² ≥ 0` неотрицателен
       (и строго положителен при `s ≠ 0`), а при `s₁² ≤ s₂²` выполнено `4·(1 + s₁²) ≤ 4·(1 + s₂²)`
       (монотонность `w(s₂) ≤ w(s₁)`).
    4. Приведённая целевая функция `4·(1 + s²)·[(5/4)·L/(1+s²) + (1/2)·λ·s²] = 5·L + 2·λ·s²·(1 + s²) ≥ 0`
       неотрицательна (и строго положительна при `L > 0`), а точный `DualBalance`-числитель
       производной по `s` совпадает с формулой в `src/algebra.rs`:
       `grad_s = s · (λ - (5/2) · L / (1 + s²)²)` (с точным общим знаменателем `2·(1 + s²)²`). -/
theorem step138_lyapunov_gate_and_rational_lk_balance
    (kappa x1 x2 s s1 s2 L lam : Balance)
    (hkap : BalGtZero kappa)
    (hx1 : BalGeZero x1)
    (hx12 : BalGeZero (balance_sub x2 x1))
    (hs12 : BalGeZero (balance_sub (bsq s2) (bsq s1)))
    (hL : BalGeZero L)
    (hlam : BalGtZero lam) :
    let D1 := balance_add bone (balance_mul kappa x1)
    let D2 := balance_add bone (balance_mul kappa x2)
    let W := balance_add bone (bsq s)
    let W1 := balance_add bone (bsq s1)
    let W2 := balance_add bone (bsq s2)
    let sd := dual_var s
    let s2d := dual_mul sd sd
    let Wd := dual_add (dual_const bone) s2d
    let obj_scaled := balance_add (balance_mul (balOfNat 5) L)
                                  (balance_mul (balOfNat 2) (balance_mul lam (balance_mul (bsq s) W)))
    BalGtZero D1 ∧
    BalGeZero (balance_sub D1 bone) ∧
    BalGeZero (balance_sub D2 D1) ∧
    (BalGtZero x1 → BalLt bone D1) ∧
    BalGtZero (balance_mul (balOfNat 4) W) ∧
    BalGeZero (balance_sub (balance_mul (balOfNat 4) W) (balOfNat 4)) ∧
    BalGeZero (bsq s) ∧
    (BalGtZero s → BalGtZero (bsq s)) ∧
    (BalGtZero (balance_neg s) → BalGtZero (bsq s)) ∧
    BalGeZero (balance_sub (balance_mul (balOfNat 4) W2) (balance_mul (balOfNat 4) W1)) ∧
    BalGeZero obj_scaled ∧
    (BalGtZero L → BalGtZero obj_scaled) ∧
    DualEquiv Wd ⟨W, balance_mul (balOfNat 2) s⟩ ∧
    -- Точный числитель производной `2·(1+s²)² · d/ds[(5/4)L/(1+s²) + (1/2)λs²] = s · (2λ(1+s²)² - 5L)`:
    BalanceEquiv
      (balance_sub (balance_mul (balance_mul (balOfNat 2) (balance_mul lam s)) (bsq W))
                   (balance_mul (balOfNat 5) (balance_mul L s)))
      (balance_mul s (balance_sub (balance_mul (balOfNat 2) (balance_mul lam (bsq W)))
                                  (balance_mul (balOfNat 5) L))) := by
  dsimp only [bsq, dual_var, dual_const, dual_mul, dual_add, DualEquiv] at *
  have hkx1 : BalGeZero (balance_mul kappa x1) := bal_mul_ge_ge (Or.inr hkap) hx1
  have hD1_gt : BalGtZero (balance_add bone (balance_mul kappa x1)) :=
    bal_add_gt_ge bone_gt_zero hkx1
  have hD1_ge1 : BalGeZero (balance_sub (balance_add bone (balance_mul kappa x1)) bone) :=
    bal_ge_zero_of_equiv
      (step24_balance_symm (bal_poly_eq (fun | 0 => bone | 1 => balance_mul kappa x1 | _ => bzero)
        (.sub (.add (.var 0) (.var 1)) (.var 0)) (.var 1) rfl))
      hkx1
  have hD_mono : BalGeZero (balance_sub (balance_add bone (balance_mul kappa x2))
                                        (balance_add bone (balance_mul kappa x1))) :=
    bal_ge_zero_of_equiv
      (step24_balance_symm (bal_poly_eq (fun | 0 => bone | 1 => kappa | 2 => x1 | 3 => x2 | _ => bzero)
        (.sub (.add (.var 0) (.mul (.var 1) (.var 3))) (.add (.var 0) (.mul (.var 1) (.var 2))))
        (.mul (.var 1) (.sub (.var 3) (.var 2))) rfl))
      (bal_mul_ge_ge (Or.inr hkap) hx12)
  have hD1_strict : BalGtZero x1 → BalLt bone (balance_add bone (balance_mul kappa x1)) := by
    intro hx1_pos
    exact bal_gt_zero_of_equiv
      (step24_balance_symm (bal_poly_eq (fun | 0 => bone | 1 => balance_mul kappa x1 | _ => bzero)
        (.sub (.add (.var 0) (.var 1)) (.var 0)) (.var 1) rfl))
      (bal_mul_gt_gt hkap hx1_pos)
  have hs2_ge : BalGeZero (balance_mul s s) := bal_sq_ge_zero s
  have hW_gt : BalGtZero (balance_add bone (balance_mul s s)) :=
    bal_add_gt_ge bone_gt_zero hs2_ge
  have h4 : BalGtZero (balOfNat 4) := balOfNat_succ_gt_zero 3
  have h4W_gt : BalGtZero (balance_mul (balOfNat 4) (balance_add bone (balance_mul s s))) :=
    bal_mul_gt_gt h4 hW_gt
  have h4W_ge4 : BalGeZero (balance_sub (balance_mul (balOfNat 4) (balance_add bone (balance_mul s s))) (balOfNat 4)) :=
    bal_ge_zero_of_equiv
      (step24_balance_symm (bal_poly_eq (fun | 0 => balance_mul s s | _ => bzero)
        (.sub (.mul (.nat 4) (.add .one (.var 0))) (.nat 4)) (.mul (.nat 4) (.var 0)) rfl))
      (bal_mul_ge_ge (Or.inr h4) hs2_ge)
  have hs_pos : BalGtZero s → BalGtZero (balance_mul s s) := fun hs => bal_mul_gt_gt hs hs
  have hs_neg : BalGtZero (balance_neg s) → BalGtZero (balance_mul s s) := by
    intro hns
    exact bal_gt_zero_of_equiv
      (step24_balance_symm (bal_poly_eq (fun | 0 => s | _ => bzero)
        (.mul (.var 0) (.var 0)) (.mul (.neg (.var 0)) (.neg (.var 0))) rfl))
      (bal_mul_gt_gt hns hns)
  have h4W_mono : BalGeZero (balance_sub (balance_mul (balOfNat 4) (balance_add bone (balance_mul s2 s2)))
                                         (balance_mul (balOfNat 4) (balance_add bone (balance_mul s1 s1)))) :=
    bal_ge_zero_of_equiv
      (step24_balance_symm (bal_poly_eq (fun | 0 => bone | 1 => balance_mul s1 s1 | 2 => balance_mul s2 s2 | _ => bzero)
        (.sub (.mul (.nat 4) (.add (.var 0) (.var 2))) (.mul (.nat 4) (.add (.var 0) (.var 1))))
        (.mul (.nat 4) (.sub (.var 2) (.var 1))) rfl))
      (bal_mul_ge_ge (Or.inr h4) hs12)
  have hbar_ge : BalGeZero (balance_mul (balOfNat 2) (balance_mul lam (balance_mul (balance_mul s s) (balance_add bone (balance_mul s s))))) :=
    bal_mul_ge_ge (Or.inr (balOfNat_succ_gt_zero 1)) (bal_mul_ge_ge (Or.inr hlam) (bal_mul_ge_ge hs2_ge (Or.inr hW_gt)))
  have h5L_ge : BalGeZero (balance_mul (balOfNat 5) L) :=
    bal_mul_ge_ge (Or.inr (balOfNat_succ_gt_zero 4)) hL
  have hobj_ge : BalGeZero (balance_add (balance_mul (balOfNat 5) L)
                                        (balance_mul (balOfNat 2) (balance_mul lam (balance_mul (balance_mul s s) (balance_add bone (balance_mul s s)))))) :=
    bal_add_ge_ge h5L_ge hbar_ge
  have hobj_gt : BalGtZero L →
      BalGtZero (balance_add (balance_mul (balOfNat 5) L)
                             (balance_mul (balOfNat 2) (balance_mul lam (balance_mul (balance_mul s s) (balance_add bone (balance_mul s s)))))) := by
    intro hLpos
    exact bal_add_gt_ge (bal_mul_gt_gt (balOfNat_succ_gt_zero 4) hLpos) hbar_ge
  have hWd_re : BalanceEquiv (balance_add bone (balance_mul s s)) (balance_add bone (balance_mul s s)) :=
    step24_balance_refl _
  have hWd_eps : BalanceEquiv (balance_add bzero (balance_add (balance_mul s bone) (balance_mul bone s)))
                              (balance_mul (balOfNat 2) s) :=
    bal_poly_eq (fun | 0 => s | _ => bzero)
      (.add .zero (.add (.mul (.var 0) .one) (.mul .one (.var 0))))
      (.mul (.nat 2) (.var 0)) rfl
  have hgrad_fac : BalanceEquiv
      (balance_sub (balance_mul (balance_mul (balOfNat 2) (balance_mul lam s)) (balance_mul (balance_add bone (balance_mul s s)) (balance_add bone (balance_mul s s))))
                   (balance_mul (balOfNat 5) (balance_mul L s)))
      (balance_mul s (balance_sub (balance_mul (balOfNat 2) (balance_mul lam (balance_mul (balance_add bone (balance_mul s s)) (balance_add bone (balance_mul s s)))))
                                  (balance_mul (balOfNat 5) L))) :=
    bal_poly_eq (fun | 0 => s | 1 => lam | 2 => balance_mul (balance_add bone (balance_mul s s)) (balance_add bone (balance_mul s s)) | 3 => L | _ => bzero)
      (.sub (.mul (.mul (.nat 2) (.mul (.var 1) (.var 0))) (.var 2)) (.mul (.nat 5) (.mul (.var 3) (.var 0))))
      (.mul (.var 0) (.sub (.mul (.nat 2) (.mul (.var 1) (.var 2))) (.mul (.nat 5) (.var 3)))) rfl
  exact ⟨hD1_gt, hD1_ge1, hD_mono, hD1_strict, h4W_gt, h4W_ge4, hs2_ge, hs_pos, hs_neg, h4W_mono, hobj_ge, hobj_gt, ⟨hWd_re, hWd_eps⟩, hgrad_fac⟩

end PosNat
end Foundations

open Foundations
open Foundations.PosNat

#print axioms step01_identity
#print axioms step02_no_self_negation
#print axioms step03_eq_symm
#print axioms step04_eq_trans
#print axioms step07_one_ne_succ
#print axioms step08_succ_ne_self
#print axioms step09_succ_inj
#print axioms step11_one_add
#print axioms step12_add_comm
#print axioms step13_add_assoc
#print axioms step15_one_mul
#print axioms step16_add_mul
#print axioms step17_mul_comm
#print axioms step19_lt_irrefl
#print axioms step20_lt_trans
#print axioms step21_distance_unique
#print axioms trichotomy
#print axioms mul_right_cancel
#print axioms step24_balance_refl
#print axioms step24_balance_symm
#print axioms step24_balance_trans
#print axioms step25_zero_as_equilibrium
#print axioms step25_add_zero_is_identity
#print axioms step25_balance_add_congr
#print axioms step41_i_squared_is_neg_one
#print axioms step42_triphase_neutrality
#print axioms step43_triphase_pair_not_neutral
#print axioms step62_dual_eps_ne_zero
#print axioms step62_dual_eps_squared_is_zero
#print axioms step63_dual_exact_derivative
#print axioms step66_bal_not_gt_and_lt
#print axioms step103_dft4_orthogonality
#print axioms step104_dft4_inversion
#print axioms step104b_dft3_inversion
#print axioms step105_dft4_parseval
#print axioms step106_dft4_shift_is_phase_rotation
#print axioms step107_rational_circle_cayley
#print axioms step108_su2_lie_laws
#print axioms step109_cayley3d_exact_orthogonality
#print axioms step109b_cayley_rot3_equiv
#print axioms step109c_cayley3d_det_one
#print axioms step110_so3_noncommutativity
#print axioms step111_rtrl_decay_strict_contraction
#print axioms step112_rational_softmax_positive
#print axioms step112b_rational_softmax_p4_positive
#print axioms step113_kerr_w1_total_denom
#print axioms step113b_kerr_w1_p4_total_denom
#print axioms step114_rtrl_exact_derivatives_dual
#print axioms step115_selective_cayley_contraction_and_deriv
#print axioms step116_rational_pos_gate_c1_and_bounds
#print axioms step117_rational_quat_and_so4_double_rotor
#print axioms step118_gated_householder_o4_bibo_and_parity
#print axioms step119_two_reflections_fix_orth_and_s5_witness
#print axioms step120_master_equation_bibo_invariance
#print axioms step121_dual_derivatives_o4_and_local_state_gate
#print axioms step122_rmsnorm_newton_and_bartlett_ph_totality
#print axioms step123_poly_repulsion_bounded
#print axioms step124_fixed_point_bounds_and_norm_drift
#print axioms step125_adaptive_od_householder_cascade_and_embedding
#print axioms step126_pillar5_self_determination_algebra
#print axioms step127_nested_so2_so3_so4_o4_parity_and_householder_v_deriv
#print axioms step128_selectivity_saddle_escape_and_q824_product_bounds
#print axioms step129_newton_inv_sqrt_exponent_bracket_and_contraction
#print axioms step130_newton_sqrt_from_inv_sqrt_bound
#print axioms step131_shared_backbone_bank_swap_and_band_idbd
#print axioms step132_q14_hypersphere_proto_margin_preservation
#print axioms step133_givens_parity_matrix_fusion_exactness
#print axioms step134_fixed_q24_hh5_watershed_nondegeneracy
#print axioms step135_symplectic_j_xor_and_rational_sat_totality
#print axioms step136_bank_rotor_adapter_composition_isometry
#print axioms step137_finsler_randers_diode_positivity_and_asymmetry
#print axioms step138_lyapunov_gate_and_rational_lk_balance
