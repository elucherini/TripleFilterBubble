"""Tests for YAML configuration loading and validation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import tempfile
from pydantic import ValidationError
from global_params import Params


class TestParamsBasicInstantiation:
    """Test basic Params instantiation (backward compatibility)."""

    def test_default_params(self):
        """Test that default params can be instantiated."""
        params = Params()
        assert params.numguys == 500
        assert params.numfriends == 20
        assert params.seed == 42

    def test_custom_params(self):
        """Test that custom params can be instantiated programmatically."""
        params = Params(
            numguys=100,
            numfriends=10,
            seed=123,
            new_info_mode="individual"
        )
        assert params.numguys == 100
        assert params.numfriends == 10
        assert params.seed == 123
        assert params.new_info_mode == "individual"

    def test_all_fields_accessible(self):
        """Test that all fields are accessible after instantiation."""
        params = Params()
        # Just verify we can access all the main fields
        assert isinstance(params.memory, int)
        assert isinstance(params.acceptance_latitude, float)
        assert isinstance(params.acceptance_sharpness, float)
        assert isinstance(params.max_pxcor, float)
        assert isinstance(params.measurement_ticks, list)


class TestYAMLLoading:
    """Test loading parameters from YAML files and strings."""

    def test_from_yaml_string_minimal(self):
        """Test loading minimal YAML config (most values default)."""
        yaml_str = """
numguys: 100
numfriends: 10
seed: 999
"""
        params = Params.from_yaml_string(yaml_str)
        assert params.numguys == 100
        assert params.numfriends == 10
        assert params.seed == 999
        # Check defaults are preserved
        assert params.memory == 20
        assert params.acceptance_latitude == 0.3

    def test_from_yaml_string_full(self):
        """Test loading complete YAML config with all fields."""
        yaml_str = """
memory: 15
acceptance_latitude: 0.4
acceptance_sharpness: 25.0
numguys: 200
numfriends: 15
network_type: "watts-strogatz"
numgroups: 3
fraction_inter: 0.3
dims: 1
birth_death_probability: 0.1
refriend_probability: 0.05
numcentral: 2
new_info_mode: "individual"
posting: false
numticks: 50
max_pxcor: 20.0
seed: 12345
run_dir: "custom_data"
measurement_ticks: [10, 25, 49]
"""
        params = Params.from_yaml_string(yaml_str)
        assert params.memory == 15
        assert params.acceptance_latitude == 0.4
        assert params.acceptance_sharpness == 25.0
        assert params.numguys == 200
        assert params.numfriends == 15
        assert params.network_type == "watts-strogatz"
        assert params.numgroups == 3
        assert params.fraction_inter == 0.3
        assert params.dims == 1
        assert params.birth_death_probability == 0.1
        assert params.refriend_probability == 0.05
        assert params.numcentral == 2
        assert params.new_info_mode == "individual"
        assert params.posting is False
        assert params.numticks == 50
        assert params.max_pxcor == 20.0
        assert params.seed == 12345
        assert params.run_dir == "custom_data"
        assert params.measurement_ticks == [10, 25, 49]

    def test_from_yaml_file(self):
        """Test loading from actual YAML file."""
        yaml_content = """
numguys: 50
numfriends: 5
seed: 777
new_info_mode: "select close infobits"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            params = Params.from_yaml(temp_path)
            assert params.numguys == 50
            assert params.numfriends == 5
            assert params.seed == 777
            assert params.new_info_mode == "select close infobits"
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_file_not_found(self):
        """Test that loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            Params.from_yaml("nonexistent_config.yaml")

    def test_load_example_config(self):
        """Test that the example config file is valid and loadable."""
        example_path = Path(__file__).parent.parent / "config" / "example_params.yaml"
        if example_path.exists():
            params = Params.from_yaml(str(example_path))
            # Verify it loaded successfully with expected defaults
            assert params.numguys == 500
            assert params.seed == 42
            assert params.new_info_mode == "central"


class TestValidation:
    """Test Pydantic validation of parameter values."""

    def test_invalid_memory_negative(self):
        """Test that negative memory is rejected."""
        with pytest.raises(ValidationError, match="memory"):
            Params(memory=0)

    def test_invalid_acceptance_latitude_negative(self):
        """Test that non-positive acceptance_latitude is rejected."""
        with pytest.raises(ValidationError, match="acceptance_latitude"):
            Params(acceptance_latitude=0.0)

    def test_invalid_acceptance_sharpness_negative(self):
        """Test that non-positive acceptance_sharpness is rejected."""
        with pytest.raises(ValidationError, match="acceptance_sharpness"):
            Params(acceptance_sharpness=-1.0)

    def test_invalid_numguys_negative(self):
        """Test that negative numguys is rejected."""
        with pytest.raises(ValidationError, match="numguys"):
            Params(numguys=-1)

    def test_invalid_numfriends_negative(self):
        """Test that negative numfriends is rejected."""
        with pytest.raises(ValidationError, match="numfriends"):
            Params(numfriends=-1)

    def test_invalid_network_type(self):
        """Test that invalid network_type is rejected."""
        with pytest.raises(ValidationError, match="network_type"):
            Params(network_type="invalid")

    def test_invalid_dims(self):
        """Test that invalid dims value is rejected."""
        with pytest.raises(ValidationError, match="dims"):
            Params(dims=3)

    def test_invalid_new_info_mode(self):
        """Test that invalid new_info_mode is rejected."""
        with pytest.raises(ValidationError, match="new_info_mode"):
            Params(new_info_mode="invalid_mode")

    def test_invalid_probability_greater_than_one(self):
        """Test that probability > 1.0 is rejected."""
        with pytest.raises(ValidationError, match="birth_death_probability"):
            Params(birth_death_probability=1.5)

    def test_invalid_probability_negative(self):
        """Test that negative probability is rejected."""
        with pytest.raises(ValidationError, match="refriend_probability"):
            Params(refriend_probability=-0.1)

    def test_invalid_fraction_inter_greater_than_one(self):
        """Test that fraction_inter > 1.0 is rejected."""
        with pytest.raises(ValidationError, match="fraction_inter"):
            Params(fraction_inter=1.5)

    def test_invalid_numticks_zero(self):
        """Test that zero or negative numticks is rejected."""
        with pytest.raises(ValidationError, match="numticks"):
            Params(numticks=0)

    def test_invalid_max_pxcor_negative(self):
        """Test that non-positive max_pxcor is rejected."""
        with pytest.raises(ValidationError, match="max_pxcor"):
            Params(max_pxcor=0.0)

    def test_extra_field_rejected(self):
        """Test that extra unknown fields are rejected."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            Params(unknown_field="value")

    def test_extra_field_in_yaml(self):
        """Test that extra fields in YAML are rejected."""
        yaml_str = """
numguys: 100
unknown_parameter: 999
"""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            Params.from_yaml_string(yaml_str)


class TestMeasurementTicksValidation:
    """Test validation of measurement_ticks field."""

    def test_measurement_ticks_empty(self):
        """Test that empty measurement_ticks is valid."""
        params = Params(measurement_ticks=[])
        assert params.measurement_ticks == []

    def test_measurement_ticks_sorted(self):
        """Test that measurement_ticks are automatically sorted."""
        params = Params(measurement_ticks=[50, 10, 25])
        assert params.measurement_ticks == [10, 25, 50]

    def test_measurement_ticks_negative_rejected(self):
        """Test that negative tick values are rejected."""
        with pytest.raises(ValidationError, match="non-negative"):
            Params(measurement_ticks=[10, -5, 20])

    def test_measurement_ticks_from_yaml(self):
        """Test that measurement_ticks load correctly from YAML."""
        yaml_str = """
measurement_ticks: [100, 50, 25]
"""
        params = Params.from_yaml_string(yaml_str)
        # Should be sorted
        assert params.measurement_ticks == [25, 50, 100]


class TestRoundTrip:
    """Test that params can be converted to dict and back."""

    def test_params_to_dict(self):
        """Test converting Params to dict."""
        params = Params(numguys=100, seed=42)
        params_dict = params.model_dump()
        assert isinstance(params_dict, dict)
        assert params_dict['numguys'] == 100
        assert params_dict['seed'] == 42

    def test_dict_to_params(self):
        """Test creating Params from dict."""
        params_dict = {
            'numguys': 100,
            'numfriends': 10,
            'seed': 42,
            'new_info_mode': 'central'
        }
        params = Params(**params_dict)
        assert params.numguys == 100
        assert params.numfriends == 10

    def test_yaml_round_trip(self):
        """Test YAML -> Params -> dict -> Params."""
        yaml_str = """
numguys: 100
numfriends: 10
seed: 999
new_info_mode: "individual"
measurement_ticks: [10, 20, 30]
"""
        params1 = Params.from_yaml_string(yaml_str)
        params_dict = params1.model_dump()
        params2 = Params(**params_dict)

        # Compare key fields
        assert params1.numguys == params2.numguys
        assert params1.numfriends == params2.numfriends
        assert params1.seed == params2.seed
        assert params1.new_info_mode == params2.new_info_mode
        assert params1.measurement_ticks == params2.measurement_ticks


class TestEdgeCases:
    """Test edge cases and boundary values."""

    def test_zero_numguys(self):
        """Test that zero numguys is allowed (edge case)."""
        params = Params(numguys=0)
        assert params.numguys == 0

    def test_zero_numfriends(self):
        """Test that zero numfriends is allowed."""
        params = Params(numfriends=0)
        assert params.numfriends == 0

    def test_probability_exactly_zero(self):
        """Test that probability = 0.0 is valid."""
        params = Params(birth_death_probability=0.0, refriend_probability=0.0)
        assert params.birth_death_probability == 0.0
        assert params.refriend_probability == 0.0

    def test_probability_exactly_one(self):
        """Test that probability = 1.0 is valid."""
        params = Params(birth_death_probability=1.0, refriend_probability=1.0)
        assert params.birth_death_probability == 1.0
        assert params.refriend_probability == 1.0

    def test_all_info_modes(self):
        """Test that all valid info modes are accepted."""
        modes = ["central", "individual", "select close infobits", "select distant infobits"]
        for mode in modes:
            params = Params(new_info_mode=mode)
            assert params.new_info_mode == mode

    def test_both_network_types(self):
        """Test that both network types are accepted."""
        params1 = Params(network_type="groups")
        assert params1.network_type == "groups"

        params2 = Params(network_type="watts-strogatz")
        assert params2.network_type == "watts-strogatz"

    def test_both_dims(self):
        """Test that both dimension values are accepted."""
        params1 = Params(dims=1)
        assert params1.dims == 1

        params2 = Params(dims=2)
        assert params2.dims == 2
