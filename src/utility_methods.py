import constants
import collections


def get_doas_from_category(category_prediction):
    doa_list = []

    for doa, category in constants.class_ids.items():
        if category == category_prediction:
            doa_list.append(float(doa))
            if float(doa) == 180.0:
                doa_list.append(float(doa))

    return doa_list


def get_possible_doas(doa):
    value = constants.class_ids.get(str(doa))
    doa_list = get_doas_from_category(value)
    return doa_list


def cylindrical(angle):
    if angle >= 360:
        angle = angle - 360
    if angle < 0:
        angle = angle + 360

    return angle


def get_quadrant(val):
    quadrant = None

    if 0 <= val < 90:
        quadrant = "first_quadrant"

    elif 90 <= val < 180:
        quadrant = "second_quadrant"
    elif 180 <= val < 270:
        quadrant = "third_quadrant"
    elif 270 <= val < 360:
        quadrant = "fourth_quadrant"

    return quadrant


def check_if_twice(prediction_list, iteration):
    flatten_list = [j for sub in prediction_list for j in sub]


    first_quadrant = len([i for i in flatten_list if 0 <= i < 90])
    second_quadrant = len([i for i in flatten_list if 90 <= i < 180])
    third_quadrant = len([i for i in flatten_list if 180 <= i < 270])
    fourth_quadrant = len([i for i in flatten_list if 270 <= i < 360])

    max_q = max([first_quadrant, second_quadrant, third_quadrant, fourth_quadrant])
    quadrants = {first_quadrant: "first_quadrant", second_quadrant: "second_quadrant",
                 third_quadrant: "third_quadrant", fourth_quadrant: "fourth_quadrant"}


    max_quadrant = None

    for key, val in quadrants.items():
        if key == max_q:
            max_quadrant = val

    counter = collections.Counter(flatten_list)


    val = None

    if 2 in counter.values():
        for key, value in counter.items():

            if value == 2 and (get_quadrant(key) == max_quadrant):
                val = key
                break

    if (val == 0.0 or val == 360.0) and iteration == 0:
        return None

    if val == 180.0 and iteration == 0:
        return None

    return val


def get_mean_prediction(prediction_list):
    prediction = None
    flatten_list = [j for sub in prediction_list for j in sub]

    first_quadrant_size = len([i for i in flatten_list if 0 <= i < 90])
    second_quadrant_size = len([i for i in flatten_list if 90 <= i < 180])
    third_quadrant_size = len([i for i in flatten_list if 180 <= i < 270])
    fourth_quadrant_size = len([i for i in flatten_list if 270 <= i < 360])

    max_quadrant = max([first_quadrant_size, second_quadrant_size, third_quadrant_size, fourth_quadrant_size])

    if max_quadrant == first_quadrant_size:
        prediction = sum([i for i in flatten_list if 0 <= i < 90]) / max_quadrant
    elif max_quadrant == second_quadrant_size:
        prediction = sum([i for i in flatten_list if 90 <= i < 180]) / max_quadrant
    elif max_quadrant == third_quadrant_size:
        prediction = sum([i for i in flatten_list if 180 <= i < 270]) / max_quadrant
    elif max_quadrant == fourth_quadrant_size:
        prediction = sum([i for i in flatten_list if 270 <= i < 360]) / max_quadrant

    return prediction
