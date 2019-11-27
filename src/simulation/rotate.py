

def get_90_deg_rotation(doa_list):
    if 90.0 < doa_list[0] <= 180.0:
        rotation = 90
    elif 90.0 > doa_list[0] >= 0.0:
        rotation = -90
    else:
        rotation = 0

    return rotation


def get_45_deg_rotation(doa_list, current_position):
    flatten_list = [j for sub in doa_list for j in sub]

    first_quadrant = len([i for i in flatten_list if 0 <= i < 90])
    second_quadrant = len([i for i in flatten_list if 90 <= i < 180])
    third_quadrant = len([i for i in flatten_list if 180 <= i < 270])
    fourth_quadrant = len([i for i in flatten_list if 270 <= i < 360])

    max_quadrant = max([first_quadrant, second_quadrant, third_quadrant, fourth_quadrant])

    if max_quadrant == first_quadrant:
        angle = 315
    elif max_quadrant == second_quadrant:
        angle = 45
    elif max_quadrant == third_quadrant:
        angle = 135
    elif max_quadrant == fourth_quadrant:
        angle = 225
    else:
        angle = 0

    rotation = angle - current_position

    return rotation


def get_fine_rotation(iteration):
    fine_rotation = 0

    if iteration == 2:
        fine_rotation = 20
    elif iteration == 3:
        fine_rotation = -40
    elif iteration == 4:
        fine_rotation = 60
    elif iteration == 5:
        fine_rotation = -80

    return fine_rotation
