def test_metadata_langkit_version():
    import whylogs as why
    from langkit import __version__
    from langkit.metadata import (
        _LANGKIT_VERSION_METADATA_KEY,
        _LANGKIT_METRIC_COLLECTION_KEY,
    )
    from langkit import light_metrics  # noqa

    expected_metric_collection_name = "light_metrics"
    text_schema = light_metrics.init()
    results = why.log({"prompt": "hello", "response": "goodbye"}, schema=text_schema)
    version = results.metadata[_LANGKIT_VERSION_METADATA_KEY]
    metric_collection_name = results.metadata[_LANGKIT_METRIC_COLLECTION_KEY]
    assert results.metadata
    assert version == __version__
    assert metric_collection_name == expected_metric_collection_name
