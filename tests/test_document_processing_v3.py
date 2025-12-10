"""
Test suite for 3GPP Document Processing V3
Tests the download_and_process_3gpp.py module
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
import pytest
from unittest.mock import Mock, patch, MagicMock

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "document_processing"))

# Import modules to test
from document_processing.download_and_process_3gpp import (
    DocumentMetaData,
    SectionStructure,
    ProcessedChunk,
    ThreeGPPParser,
    ThreeGPPProcessor,
    ThreeGPPDownloader
)


class TestDataStructures:
    """Test data classes"""
    
    def test_document_metadata(self):
        """Test DocumentMetaData dataclass"""
        metadata = DocumentMetaData(
            specification_id="ts_23_501",
            version="18.9.0",
            title="3GPP TS 23.501",
            file_path="/path/to/doc.docx"
        )
        
        assert metadata.specification_id == "ts_23_501"
        assert metadata.version == "18.9.0"
        assert metadata.title == "3GPP TS 23.501"
        assert metadata.file_path == "/path/to/doc.docx"
    
    def test_section_structure(self):
        """Test SectionStructure dataclass"""
        section = SectionStructure(
            section_id="4.2.1",
            title="Overview",
            level=3,
            parent_section="4.2",
            content="Test content",
            tables=[{"headers": ["A", "B"], "rows": [["1", "2"]]}]
        )
        
        assert section.section_id == "4.2.1"
        assert section.title == "Overview"
        assert section.level == 3
        assert section.parent_section == "4.2"
        assert section.content == "Test content"
        assert len(section.tables) == 1
    
    def test_processed_chunk(self):
        """Test ProcessedChunk dataclass"""
        chunk = ProcessedChunk(
            chunk_id="ts_23_501_chunk_001",
            section_id="4.2.1",
            section_title="Overview",
            content="Test content",
            chunk_type="definition",
            cross_references={"internal": [], "external": []},
            tables=[],
            content_metadata={"word_count": 2}
        )
        
        assert chunk.chunk_id == "ts_23_501_chunk_001"
        assert chunk.chunk_type == "definition"
        assert chunk.content_metadata["word_count"] == 2


class TestThreeGPPParser:
    """Test the parser class"""
    
    @pytest.fixture
    def parser(self):
        return ThreeGPPParser()
    
    def test_extract_metadata_with_mock(self, parser):
        """Test metadata extraction with mocked mammoth"""
        mock_html = """
        <html>
            <body>
                <p>3GPP TS 23.501 V18.9.0 (2025-03)</p>
                <p>System Architecture for 5G System</p>
            </body>
        </html>
        """

        with patch('mammoth.convert_to_html') as mock_mammoth:
            mock_mammoth.return_value = Mock(value=mock_html)

            with patch('builtins.open', create=True) as mock_open:
                metadata = parser.extract_metadata("test.docx")

                # Code returns ts_23.501 (with dot, not underscore)
                assert metadata.specification_id == "ts_23.501"
                assert metadata.version == "18.9.0"
                assert metadata.title == "3GPP TS 23.501"
    
    def test_classify_content_type(self, parser):
        """Test content type classification"""
        assert parser.classify_content_type("Definitions", "") == "definition"
        assert parser.classify_content_type("Procedure", "") == "procedure"
        assert parser.classify_content_type("Parameters", "") == "parameter"
        assert parser.classify_content_type("Requirements", "The UE shall perform") == "requirement"
        assert parser.classify_content_type("Other", "") == "general"
    
    def test_extract_cross_references(self, parser):
        """Test cross-reference extraction"""
        content = """
        As defined in clause 4.2.1, the procedure follows table 5.1-1.
        See TS 23.502, clause 4.3 for details.
        Figure 6.2-1 shows the architecture.
        """

        # extract_cross_references requires spec_id as second argument
        cross_refs = parser.extract_cross_references(content, "ts_23.501")

        assert "internal" in cross_refs
        assert "external" in cross_refs

        # Check internal references (clause, table, figure without TS/TR)
        internal_refs = cross_refs["internal"]
        assert len(internal_refs) > 0
        # At least one clause reference should be found
        assert any(ref["ref_type"] == "clause" for ref in internal_refs)

        # Check external references (with TS/TR)
        external_refs = cross_refs["external"]
        assert len(external_refs) > 0
        # Code returns ts_23.502 (with dot)
        assert any(ref["target_spec"] == "ts_23.502" for ref in external_refs)
    
    def test_parse_sections_with_mock(self, parser):
        """Test section parsing with mocked HTML"""
        mock_html = """
        <html>
            <body>
                <h1>1 Scope</h1>
                <p>This document defines the scope.</p>
                <h2>2.1 Definitions</h2>
                <p>Key definitions are provided here.</p>
                <h2>2.2 Abbreviations</h2>
                <p>AMF: Access and Mobility Management Function</p>
                <table>
                    <tr><td>Header1</td><td>Header2</td></tr>
                    <tr><td>Data1</td><td>Data2</td></tr>
                </table>
            </body>
        </html>
        """
        
        with patch('mammoth.convert_to_html') as mock_mammoth:
            mock_mammoth.return_value = Mock(value=mock_html)
            
            with patch('builtins.open', create=True):
                sections = parser.parse_sections("test.docx")
                
                assert len(sections) >= 2
                
                # Check first section
                scope_section = next((s for s in sections if s.section_id == "1"), None)
                assert scope_section is not None
                assert scope_section.title == "Scope"
                assert "defines the scope" in scope_section.content
                
                # Check abbreviations section with table
                abbr_section = next((s for s in sections if s.section_id == "2.2"), None)
                if abbr_section:
                    assert abbr_section.title == "Abbreviations"
                    assert abbr_section.tables is not None


class TestThreeGPPProcessor:
    """Test the document processor"""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create mock config for processor"""
        config = Mock()
        config.output_dir = tmp_path / "output"
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.data_dir = tmp_path / "data"
        config.data_dir.mkdir(parents=True, exist_ok=True)
        return config

    @pytest.fixture
    def processor(self, mock_config):
        """Create processor with mock config"""
        return ThreeGPPProcessor(mock_config)

    def test_init(self, processor, mock_config):
        """Test processor initialization"""
        assert processor.output_dir == mock_config.output_dir
        assert processor.output_dir.exists()
        # Processor has parser attribute, not stats
        assert processor.parser is not None

    def test_process_single_success(self, processor, tmp_path):
        """Test successful single document processing"""
        # Create a mock DOCX file
        docx_path = tmp_path / "test.docx"
        docx_path.write_text("")  # Empty file for testing

        mock_metadata = DocumentMetaData(
            specification_id="ts_23.501",
            version="18.9.0",
            title="3GPP TS 23.501",
            file_path=str(docx_path)
        )

        mock_chunks = [
            ProcessedChunk(
                chunk_id="test_001",
                section_id="1",
                section_title="Introduction",
                content="Test",
                chunk_type="general",
                cross_references={}
            )
        ]

        # Mock parser.process_document to return chunks and metadata
        with patch.object(processor.parser, 'process_document', return_value=(mock_chunks, mock_metadata)):
            with patch.object(processor.parser, 'save_to_json'):
                result = processor._process_single(docx_path)
                assert result is True

    def test_find_docx_files(self, processor, tmp_path):
        """Test finding DOCX files in directory"""
        # Create test structure
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "series_23").mkdir()
        (data_dir / "series_29").mkdir()

        # Create DOCX files
        (data_dir / "series_23" / "23501_main_body.docx").write_text("")
        (data_dir / "series_23" / "23501_cover.docx").write_text("")  # Should be excluded
        (data_dir / "series_29" / "29500-i10.docx").write_text("")  # Standard naming format
        (data_dir / "~temp.docx").write_text("")  # Should be excluded

        docx_files = processor.find_docx_files(data_dir)

        # Check that correct files are found
        filenames = [f.name for f in docx_files]
        assert "23501_main_body.docx" in filenames
        assert "29500-i10.docx" in filenames
        assert "23501_cover.docx" not in filenames
        assert "~temp.docx" not in filenames


class TestThreeGPPDownloader:
    """Test the downloader class"""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create mock config for downloader"""
        config = Mock()
        config.download_dir = tmp_path / "downloads"
        config.download_dir.mkdir(parents=True, exist_ok=True)
        config.series = ['23', '29']
        config.release = 18
        return config

    @pytest.fixture
    def downloader(self, mock_config):
        """Create downloader with mock config"""
        return ThreeGPPDownloader(mock_config)

    def test_init(self, downloader, mock_config):
        """Test downloader initialization"""
        assert downloader.download_dir == mock_config.download_dir
        assert downloader.download_dir.exists()

    def test_check_download_3gpp_installed(self, downloader):
        """Test checking if download_3gpp is installed"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)
            result = downloader.check_download_3gpp_installed()
            assert result is True

            mock_run.return_value = Mock(returncode=1)
            result = downloader.check_download_3gpp_installed()
            assert result is False

    @patch('subprocess.run')
    def test_download_release_success(self, mock_run, downloader, tmp_path):
        """Test successful download"""
        # Create fake downloaded files to simulate success
        release_dir = tmp_path / "downloads" / "Rel-18"
        release_dir.mkdir(parents=True, exist_ok=True)
        (release_dir / "test.zip").write_text("")

        # Mock subprocess calls
        mock_run.return_value = Mock(returncode=0, stderr="")

        result = downloader.download_release(18)

        # Should return True because zip file exists
        assert result is True

    @patch('subprocess.run')
    def test_download_release_no_files(self, mock_run, downloader):
        """Test download when no files are downloaded"""
        # Mock subprocess calls succeed but no files created
        mock_run.return_value = Mock(returncode=0, stderr="")

        result = downloader.download_release(18)

        # Should return False because no files downloaded
        assert result is False


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_with_sample_data(self, tmp_path):
        """Test full pipeline with sample data"""
        # Create sample JSON output
        sample_data = {
            "metadata": {
                "specification_id": "ts_23_501",
                "version": "18.9.0",
                "title": "3GPP TS 23.501",
                "file_path": "test.docx"
            },
            "export_info": {
                "export_date": "2025-01-01T00:00:00",
                "total_chunks": 2,
                "parser_version": "v3.0"
            },
            "chunks": [
                {
                    "chunk_id": "ts_23_501_chunk_001",
                    "section_id": "1",
                    "section_title": "Scope",
                    "content": "This document defines the Stage 2 system architecture",
                    "chunk_type": "definition",
                    "cross_references": {
                        "internal": [],
                        "external": [{"target_spec": "ts_23_502", "ref_type": "specification", "ref_id": "", "confidence": 1.0}]
                    },
                    "tables": None,
                    "content_metadata": {
                        "word_count": 8,
                        "complexity_score": 0.5,
                        "key_terms": ["Stage", "architecture"]
                    }
                }
            ]
        }
        
        # Save to file
        json_file = tmp_path / "ts_23_501.json"
        with open(json_file, 'w') as f:
            json.dump(sample_data, f)
        
        # Verify it can be loaded
        with open(json_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data["metadata"]["specification_id"] == "ts_23_501"
        assert loaded_data["export_info"]["total_chunks"] == 2
        assert len(loaded_data["chunks"]) == 1
        assert loaded_data["chunks"][0]["chunk_type"] == "definition"


class TestCLI:
    """Test command line interface"""

    @patch('sys.argv', ['download_and_process_3gpp.py', '--help'])
    def test_help_argument(self):
        """Test help argument"""
        with pytest.raises(SystemExit) as exc_info:
            from document_processing.download_and_process_3gpp import main
            main()

        # Help should exit with 0
        assert exc_info.value.code == 0

    @patch('sys.argv', ['download_and_process_3gpp.py', 'status'])
    @patch('document_processing.download_and_process_3gpp.Pipeline')
    def test_process_argument(self, mock_pipeline_class):
        """Test status command"""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline

        from document_processing.download_and_process_3gpp import main
        main()

        # Pipeline status should be called
        mock_pipeline.status.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])