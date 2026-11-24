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
