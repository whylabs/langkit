import textstat
from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import register_metric_udf


@register_metric_udf(col_type=String)
def flesch_kincaid_grade(text: str) -> float:
    return textstat.textstat.flesch_kincaid_grade(text)


@register_metric_udf(col_type=String)
def flesch_reading_ease(text: str) -> float:
    return textstat.textstat.flesch_reading_ease(text)


@register_metric_udf(col_type=String)
def smog_index(text: str) -> float:
    return textstat.textstat.smog_index(text)


@register_metric_udf(col_type=String)
def coleman_liau_index(text: str) -> float:
    return textstat.textstat.coleman_liau_index(text)


@register_metric_udf(col_type=String)
def automated_readability_index(text: str) -> float:
    return textstat.textstat.automated_readability_index(text)


@register_metric_udf(col_type=String)
def dale_chall_readability_score(text: str) -> float:
    return textstat.textstat.dale_chall_readability_score(text)


@register_metric_udf(col_type=String)
def difficult_words(text: str) -> float:
    return textstat.textstat.difficult_words(text)


@register_metric_udf(col_type=String)
def linsear_write_formula(text: str) -> float:
    return textstat.textstat.linsear_write_formula(text)


@register_metric_udf(col_type=String)
def gunning_fog(text: str) -> float:
    return textstat.textstat.gunning_fog(text)


@register_metric_udf(col_type=String)
def aggregate_reading_level(text: str) -> float:
    return textstat.textstat.text_standard(text)


@register_metric_udf(col_type=String)
def fernandez_huerta(text: str) -> float:
    return textstat.textstat.fernandez_huerta(text)


@register_metric_udf(col_type=String)
def szigriszt_pazos(text: str) -> float:
    return textstat.textstat.szigriszt_pazos(text)


@register_metric_udf(col_type=String)
def gutierrez_polini(text: str) -> float:
    return textstat.textstat.gutierrez_polini(text)


@register_metric_udf(col_type=String)
def crawford(text: str) -> float:
    return textstat.textstat.crawford(text)


@register_metric_udf(col_type=String)
def gulpease_index(text: str) -> float:
    return textstat.textstat.gulpease_index(text)


@register_metric_udf(col_type=String)
def osman(text: str) -> float:
    return textstat.textstat.osman(text)
