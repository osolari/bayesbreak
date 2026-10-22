from scripts.check_annotations import check_annotations


def test_each_bibliographic_reference_has_a_substantive_annotation():
    result=check_annotations()
    assert result['passed'], result
    assert result['counts']=={'bibliography_keys':38,'annotation_files':38}
