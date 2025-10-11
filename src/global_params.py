from dataclasses import dataclass

@dataclass
class Params:
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
    memory: int = 20
    acceptance_latitude: float = 0.3      # lambda
    acceptance_sharpness: float = 20.0    # k
    numguys: int = 500
    numfriends: int = 20
    network_type: str = "groups"          # "groups" or "watts-strogatz"
    numgroups: int = 4
    fraction_inter: float = 0.2           # inter-group target fraction
    dims: int = 2                         # 1 or 2
    birth_death_probability: float = 0.0
    refriend_probability: float = 0.0
    numcentral: int = 1
    new_info_mode: str = "central"        # "central" | "individual" | "select close infobits" | "select distant infobits"
    posting: bool = True
    # numticks: int = 10_000
    numticks: int = 100
    plot_update_every: int = 20

    # world geometry (NetLogo uses patch coords; we emulate)
    max_pxcor: float = 16.0               # half-width/height of world (so coords ∈ [-max_pxcor, max_pxcor])
    quantize: bool = True
    quantization_scale: float = 65535.0

    seed: int = 42

    run_dir: str = "data"