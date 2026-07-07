"""Единый конфиг скоринга и порогов бизнес-отчёта.

Раньше «магические числа» (границы шума вокруг среднего, пороги концентрации,
веса score, лимиты отбора) были размазаны по детекторам `mining.py`. Здесь —
ОДИН источник правды: пороги легко ревьюить/тюнить, а часть из них адаптируется
к данным (`min_rows` от размера таблицы). Значения совпадают с прежними
инлайн-константами, поэтому централизация не меняет поведение детекторов.

Плюс здесь же — БАЗОВЫЙ РАНГ РАЗДЕЛОВ и разметка «discovery»-разделов: их порядок
в отчёте определяется значимостью (score находок), а не фиксированным списком.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    # --- материальность / объём среза ---
    min_rows_abs: int = 30
    min_rows_frac: float = 0.005
    # --- rate/duration: что считаем «шумом» вокруг среднего (не интересно) ---
    rate_abs_gap: float = 0.05          # |отклонение| меньше — игнор
    rate_lift_lo: float = 0.77          # 0.77..1.3× среднего — в пределах нормы
    rate_lift_hi: float = 1.3
    dur_lift_lo: float = 0.7            # сроки: 0.7..1.43× — норма
    dur_lift_hi: float = 1.43
    # --- концентрация (Парето) ---
    conc_top_share: float = 0.25        # лидер < 25% И …
    conc_n80_frac: float = 0.35         # … 80% дают > 35% категорий — не концентрировано
    # --- перекос ценность/количество ---
    mismatch_w_share: float = 0.08      # доля веса среза не ниже
    mismatch_ratio: float = 1.8         # ценность/количество не ниже
    # --- 2D-взаимодействия ---
    inter_hi: float = 1.4               # ≥ — сильнее ожидаемого
    inter_lo: float = 0.6               # ≤ — слабее ожидаемого
    inter_material: float = 0.01        # ниже И мало строк — отбрасываем
    # --- отбор находок ---
    select_total: int = 14
    kind_cap_rate: int = 6
    kind_cap_inter: int = 3
    kind_cap_conc: int = 4
    kind_cap_mismatch: int = 3
    # --- веса score ---
    w_money_boost: float = 2.0          # денежный срез весомее
    w_inter: float = 2.5
    w_mismatch: float = 4.0
    w_conc_top: float = 3.0

    def min_rows(self, n: int) -> int:
        """Порог объёма среза, адаптивный к размеру таблицы."""
        return max(self.min_rows_abs, int(n * self.min_rows_frac))

    @property
    def kind_cap(self) -> dict[str, int]:
        return {"rate_dev": self.kind_cap_rate, "interaction": self.kind_cap_inter,
                "money_conc": self.kind_cap_conc, "value_mismatch": self.kind_cap_mismatch}


TH = Thresholds()


# Базовый ранг разделов (больше — раньше). Разделы-«находки» (discovery) делят один
# ранг и внутри него упорядочиваются по значимости (агрегатный score находок),
# поэтому самый острый разрез всплывает выше. Фокус/обзор — всегда сверху,
# сущности/время — снизу как контекст.
SECTION_BASE_RANK: dict[str, int] = {
    "🎯 Ответ на ваш запрос": 100,
    "📈 Обзор по показателям": 90,
    "💰 Где сосредоточены деньги и объёмы": 70,
    "⚖️ Ценность важнее количества": 70,
    "🚨 Аномальные срезы": 70,
    "👥 Ключевые игроки: сотрудники, клиенты, ИНН": 40,
    "🔍 Закономерности во времени": 20,
}

DISCOVERY_SECTIONS: set[str] = {
    "💰 Где сосредоточены деньги и объёмы",
    "⚖️ Ценность важнее количества",
    "🚨 Аномальные срезы",
}


def section_sort_key(header: str, agg_score: float) -> tuple[int, float]:
    """Ключ сортировки раздела: базовый ранг, затем (для discovery) значимость находок.
    Не-discovery разделы держат позицию базовым рангом; их score-компонента нейтральна."""
    base = SECTION_BASE_RANK.get(header, 50)
    tie = agg_score if header in DISCOVERY_SECTIONS else 0.0
    return (base, tie)
