"""
Tests for SubjectClassifier - Subject-based KG Enhancement.
"""
import pytest
import sys
sys.path.insert(0, '/home/linguyen/3GPP')

from subject_classifier import SubjectClassifier, Subject, SubjectClassification


class TestSubjectClassifier:
    """Test SubjectClassifier chunk classification"""

    @pytest.fixture
    def classifier(self):
        return SubjectClassifier()

    # Test chunk classification
    def test_classify_abbreviation_chunk(self, classifier):
        """Abbreviation chunks should be classified as Lexicon"""
        chunk = {
            'chunk_id': 'test_1',
            'section_title': 'Abbreviations',
            'content': 'AMF: Access and Mobility Management Function',
            'chunk_type': 'abbreviation',
            'spec_id': 'TS_23.501'
        }
        result = classifier.classify_chunk(chunk)
        assert result.subject == Subject.LEXICON
        assert result.confidence >= 0.9

    def test_classify_definition_chunk(self, classifier):
        """Definition chunks should be classified as Lexicon"""
        chunk = {
            'chunk_id': 'test_2',
            'section_title': 'Definitions',
            'content': 'Network Function: A functional building block',
            'chunk_type': 'definition',
            'spec_id': 'TS_23.501'
        }
        result = classifier.classify_chunk(chunk)
        assert result.subject == Subject.LEXICON

    def test_classify_procedure_chunk(self, classifier):
        """Procedure chunks should be classified as Standards specifications"""
        chunk = {
            'chunk_id': 'test_3',
            'section_title': '4.2.2 Registration procedure',
            'content': 'The registration procedure is used when...',
            'chunk_type': 'procedure',
            'spec_id': 'TS_23.502'
        }
        result = classifier.classify_chunk(chunk)
        assert result.subject == Subject.STANDARDS_SPECIFICATIONS

    def test_classify_overview_chunk(self, classifier):
        """Overview chunks should be classified as Standards overview"""
        chunk = {
            'chunk_id': 'test_4',
            'section_title': 'System architecture overview',
            'content': 'The 5G system architecture consists of...',
            'chunk_type': 'overview',
            'spec_id': 'TS_23.501'
        }
        result = classifier.classify_chunk(chunk)
        assert result.subject == Subject.STANDARDS_OVERVIEW

    def test_classify_3gpp_default(self, classifier):
        """3GPP spec chunks default to Standards specifications"""
        chunk = {
            'chunk_id': 'test_5',
            'section_title': 'Some section',
            'content': 'Some technical content',
            'chunk_type': 'technical',
            'spec_id': 'TS_23.501'
        }
        result = classifier.classify_chunk(chunk)
        assert result.subject == Subject.STANDARDS_SPECIFICATIONS


class TestQuerySubjectDetection:
    """Test query subject detection"""

    @pytest.fixture
    def classifier(self):
        return SubjectClassifier()

    def test_detect_lexicon_query(self, classifier):
        """Definition queries should expect Lexicon subject"""
        queries = [
            "What does SCP stand for?",
            "What is the definition of AMF?",
            "What is the meaning of UPF?"
        ]
        for query in queries:
            subjects = classifier.detect_query_subjects(query)
            subject_names = [s.value for s, w in subjects]
            assert "Lexicon" in subject_names

    def test_detect_standards_spec_query(self, classifier):
        """Procedure queries should expect Standards specifications"""
        queries = [
            "What is the registration procedure in 3GPP Release 17?",
            "How does the NAS message work?",
            "Explain the RRC procedure"
        ]
        for query in queries:
            subjects = classifier.detect_query_subjects(query)
            subject_names = [s.value for s, w in subjects]
            assert "Standards specifications" in subject_names

    def test_detect_research_query(self, classifier):
        """Algorithm queries should expect Research publications"""
        queries = [
            "What algorithm is used for beamforming optimization?",
            "Explain the machine learning technique for resource allocation"
        ]
        for query in queries:
            subjects = classifier.detect_query_subjects(query)
            subject_names = [s.value for s, w in subjects]
            assert "Research publications" in subject_names

    def test_detect_overview_query(self, classifier):
        """General 'how does' queries should expect Standards overview"""
        queries = [
            "How does AMF work?",
            "What is the role of SMF?"
        ]
        for query in queries:
            subjects = classifier.detect_query_subjects(query)
            # Should return some subjects
            assert len(subjects) > 0


class TestSubjectBoost:
    """Test subject boost calculation"""

    @pytest.fixture
    def classifier(self):
        return SubjectClassifier()

    def test_boost_matching_subject(self, classifier):
        """Matching subject should return boost > 1.0"""
        expected = [(Subject.LEXICON, 1.5), (Subject.STANDARDS_SPECIFICATIONS, 1.2)]
        boost = classifier.get_subject_boost("Lexicon", expected)
        assert boost == 1.5

    def test_boost_secondary_subject(self, classifier):
        """Secondary matching subject should return its weight"""
        expected = [(Subject.LEXICON, 1.5), (Subject.STANDARDS_SPECIFICATIONS, 1.2)]
        boost = classifier.get_subject_boost("Standards specifications", expected)
        assert boost == 1.2

    def test_no_boost_non_matching(self, classifier):
        """Non-matching subject should return 1.0"""
        expected = [(Subject.LEXICON, 1.5)]
        boost = classifier.get_subject_boost("Research publications", expected)
        assert boost == 1.0

    def test_no_boost_empty_expected(self, classifier):
        """Empty expected subjects should return 1.0"""
        boost = classifier.get_subject_boost("Lexicon", [])
        assert boost == 1.0


class TestBatchClassification:
    """Test batch classification"""

    @pytest.fixture
    def classifier(self):
        return SubjectClassifier()

    def test_classify_batch(self, classifier):
        """Should classify multiple chunks"""
        chunks = [
            {'chunk_id': 'c1', 'section_title': 'Abbreviations', 'content': 'AMF', 'chunk_type': 'abbreviation', 'spec_id': 'TS_23.501'},
            {'chunk_id': 'c2', 'section_title': 'Procedure', 'content': 'Registration', 'chunk_type': 'procedure', 'spec_id': 'TS_23.502'},
        ]
        results = classifier.classify_batch(chunks)
        assert len(results) == 2
        assert 'c1' in results
        assert 'c2' in results
        assert results['c1'].subject == Subject.LEXICON
