import numpy as np


def jyperbeam_to_jyperpixel(data, header):
    """Converts image units from Jy/beam to Jy/pixel."""
    assert 'BMAJ' in header and 'BMIN' in header, "Header must contain BMAJ and BMIN keywords."
    assert 'CDELT1' in header and 'CDELT2' in header, "Header must contain CDELT1 and CDELT2 keywords."
    
    bmaj = header['BMAJ']  # Beam major axis in degrees
    bmin = header['BMIN']  # Beam minor axis in degrees

    # Convert beam size from degrees to radians
    bmaj_rad = np.deg2rad(bmaj)
    bmin_rad = np.deg2rad(bmin)

    # Calculate beam area in steradians
    beam_area_sr = (np.pi * bmaj_rad * bmin_rad) / (4 * np.log(2))

    # Get pixel scale from header (CDELT1 and CDELT2 in degrees)
    cdelt1 = header['CDELT1']
    cdelt2 = header['CDELT2']

    # Convert pixel scale from degrees to radians
    pixel_scale_sr = np.abs(np.deg2rad(cdelt1) * np.deg2rad(cdelt2))

    # Conversion factor from Jy/beam to Jy/pixel
    conversion_factor = pixel_scale_sr / beam_area_sr

    # Apply conversion
    converted_data = data * conversion_factor

    # Create the new header
    new_header = header.copy()
    new_header['BUNIT'] = 'JY/PIXEL'

    return converted_data, new_header

def jyperpixel_to_jyperbeam(data, header):
    """Converts image units from Jy/pixel to Jy/beam."""
    assert 'BMAJ' in header and 'BMIN' in header, "Header must contain BMAJ and BMIN keywords."
    assert 'CDELT1' in header and 'CDELT2' in header, "Header must contain CDELT1 and CDELT2 keywords."
    
    bmaj = header['BMAJ']  # Beam major axis in degrees
    bmin = header['BMIN']  # Beam minor axis in degrees

    # Convert beam size from degrees to radians
    bmaj_rad = np.deg2rad(bmaj)
    bmin_rad = np.deg2rad(bmin)

    # Calculate beam area in steradians
    beam_area_sr = (np.pi * bmaj_rad * bmin_rad) / (4 * np.log(2))

    # Get pixel scale from header (CDELT1 and CDELT2 in degrees)
    cdelt1 = header['CDELT1']
    cdelt2 = header['CDELT2']

    # Convert pixel scale from degrees to radians
    pixel_scale_sr = np.abs(np.deg2rad(cdelt1) * np.deg2rad(cdelt2))

    # Conversion factor from Jy/pixel to Jy/beam
    conversion_factor = beam_area_sr / pixel_scale_sr

    # Apply conversion
    converted_data = data * conversion_factor

    # Create the new header
    new_header = header.copy()
    new_header['BUNIT'] = 'JY/BEAM'
    return converted_data, new_header