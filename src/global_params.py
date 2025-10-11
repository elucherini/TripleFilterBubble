from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Literal


class Params(BaseModel):
    """
    Configuration parameters for TripleFilterBubble simulation.

    Can be instantiated programmatically or loaded from YAML using from_yaml().
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # visualization toggles (kept for parity; not drawing here)
    show_people: bool = True
    show_infobits: bool = False
    infobit_size: bool = False
    show_infolinks: bool = True
    show_friend_links: bool = False
    show_infosharer_links: bool = False
    patch_color: str = "white"
    color_axis_max: float = 0.05

    # core model
    memory: int = Field(default=20, ge=1)
    acceptance_latitude: float = Field(default=0.3, gt=0.0)  # lambda
    acceptance_sharpness: float = Field(default=20.0, gt=0.0)  # k
    numguys: int = Field(default=500, ge=0)
    numfriends: int = Field(default=20, ge=0)
    network_type: Literal["groups", "watts-strogatz"] = "groups"
    numgroups: int = Field(default=4, ge=1)
    fraction_inter: float = Field(default=0.2, ge=0.0, le=1.0)  # inter-group target fraction
    dims: Literal[1, 2] = 2
    birth_death_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    refriend_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    numcentral: int = Field(default=1, ge=1)
    new_info_mode: Literal["central", "individual", "select close infobits", "select distant infobits"] = "central"
    posting: bool = True
    numticks: int = Field(default=100, ge=1)
    plot_update_every: int = Field(default=20, ge=1)
    plot_every_n_ticks: int = Field(default=0, ge=0)  # 0 disables plotting, >0 plots every n ticks

    # world geometry (NetLogo uses patch coords; we emulate)
    max_pxcor: float = Field(default=16.0, gt=0.0)  # half-width/height of world (so coords ∈ [-max_pxcor, max_pxcor])
    quantize: bool = True
    quantization_scale: float = Field(default=65535.0, gt=0.0)

    seed: int = 42

    run_dir: str = "data"

    # Measurement configuration
    measurement_ticks: list[int] = Field(default_factory=list)  # Ticks at which to compute metrics (e.g., [10, 50, 100])

    @field_validator("measurement_ticks")
    @classmethod
    def validate_measurement_ticks(cls, v: list[int]) -> list[int]:
        """Ensure measurement_ticks are non-negative and sorted."""
        if any(tick < 0 for tick in v):
            raise ValueError("measurement_ticks must contain non-negative integers")
        return sorted(v)

    @classmethod
    def from_yaml(cls, path: str) -> "Params":
        """
        Load parameters from a YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Validated Params instance

        Example:
            params = Params.from_yaml("config/my_experiment.yaml")
        """
        import yaml
        from pathlib import Path

        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    @classmethod
    def from_yaml_string(cls, yaml_string: str) -> "Params":
        """
        Load parameters from a YAML string (useful for testing).

        Args:
            yaml_string: YAML configuration as a string

        Returns:
            Validated Params instance
        """
        import yaml

        config_dict = yaml.safe_load(yaml_string)
        return cls(**config_dict)