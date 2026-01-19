import numpy as np

MAX_X = 30
MAX_Y = 40

TIER_1_BASE = {"routing_configuration": [0, -1, +2, 3], 
               "base_coordinates": [(1,2), (20, 15), (12, 29), (30, 39)], 
               "fixture_types":["F6", "F1", "F1", "F6"], 
               "cable_length": "2m"}

TIER_2_BASE = {"routing_configuration": [0, -1, +2, +4, -3, -5, 6], 
               "base_coordinates": [(1,2), (4,19), (4,33), (13,18), (12,29), (20,15), (30,39)], 
               "fixture_types":["F6", "F2", "F1", "F1", "F1", "F1", "F6"], 
               "cable_length": "2m"}

TIER_3_BASE = {"routing_configuration": [0, -1, +2, +7, +6, +3, +4, +8, 9, -5, 10], 
               "base_coordinates": [(1,2), (4,19), (4,33), (13,18), (12,29), (19,4), (20,15), (20,25), (21,36), (30,15), (30,2)], 
               "fixture_types":["F6", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F3", "F6"], 
               "cable_length": "2m"}

TIER_4_BASE = {"routing_configuration": [0, -1, +2, -3, +4, -3, -7, +8, -7, +9, 10, -6, -5, 11], 
               "base_coordinates": [(1,2), (4,19), (4,33), (13,18), (12,29), (19,4), (20,15), (20,25), (21,36), (28,31), (30,15), (30,2)], 
               "fixture_types":["F6", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F1", "F3", "F6"],
               "cable_length": "3m"}

TIER_5_BASE = {"routing_configuration": [0, -1, +2, 3, -4, +5, -4, -7, +8, -7, -6, -10, 11, +12, -13, -9, -13, 14], 
               "base_coordinates": [(1,2), (4,19), (4,33), (11,11), (13,18), (12,29), (19,4), (20,15), (20,25), (21,36), (27,6), (30,15), (25,23), (28,31), (30,40)], 
               "fixture_types":["F6", "F2", "F1", "F3", "F1", "F1", "F2", "F1", "F1", "F1", "F1", "F3", "F1", "F1", "F4"],
               "cable_length": "3m"}


TIER_6_BASE = {"routing_configuration": [0, -1, +2, -3, 4, +6, -5, +6, 7, +11, -10, +9, -10, +11, -14, -15, +16, -15, 13, +12, +8, 17], 
               "base_coordinates": [(1,2), (4,19), (4,33), (9,4), (11,11), (13,18), (12,29), (11,39), (19,4), (20,15), (20,25), (21,36), (27,6), (30,15), (25,23), (28,31), (29,39), (24,11)], 
               "fixture_types":["F6", "F2", "F1", "F1", "F3", "F1", "F1", "F2", "F1", "F1", "F3", "F2", "F1", "F1", "F1", "F1", "F3", "F1", "F1", "F1", "F4"],
               "cable_length": "3m"}

ALL_BASES = [TIER_1_BASE, TIER_2_BASE, TIER_3_BASE, TIER_4_BASE, TIER_5_BASE, TIER_6_BASE]

RANDOM_OFFSET = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

def get_combined_coordinates():
    combined_coordinates = []
    for base in ALL_BASES:
        combined_coordinates.extend(base["base_coordinates"])

    combined_coordinates = list(set(combined_coordinates))
    return combined_coordinates

def assign_offset_to_coordinates():
    offset_map = {}
    for coord in get_combined_coordinates():
        offset = RANDOM_OFFSET[np.random.randint(0, len(RANDOM_OFFSET))]
        offset_map[coord] = offset
    return offset_map

def get_offset_coordinates():
    offset_map = assign_offset_to_coordinates()
    offset_coordinates = []
    overall_actual_offsets = []
    for base in ALL_BASES:
        offset_coords = []
        actual_offsets = []

        for coord in base["base_coordinates"]:
            offset = offset_map[coord]
            offset_coord = (max(1, min(coord[0] + offset[0], MAX_X)), max(1, min(coord[1] + offset[1], MAX_Y)))
            actual_offset = (offset_coord[0] - coord[0], offset_coord[1] - coord[1])
            actual_offsets.append(actual_offset)
            offset_coords.append(offset_coord)
        offset_coordinates.append(offset_coords)
        overall_actual_offsets.append(actual_offsets)

    return offset_coordinates, overall_actual_offsets



if __name__ == "__main__":
    pass