
# Skeleton reader for GOES-16 ABI bands: placeholder for agent to implement
# Expected capabilities:
# - open band files, read radiances/BT, handle scaling
# - reproject to target grid/projection
# - temporal collocation to SEVIR event timeline
#
# Implementations depend on file format (NetCDF) and geo libraries (xarray, pyproj, rioxarray).
# Keep here as an interface stub.
class GOESABIReader:
    def __init__(self): ...
    def read_band(self, path: str): ...
