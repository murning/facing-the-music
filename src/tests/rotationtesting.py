import utility_methods
import rotate

if __name__ == '__main__':
    doa_list = [[45, 315], [135, 45]]
    print(doa_list)
    iteration = 0
    current_position = 0

    #   ------------------------------------------------------------------------------------------------------------------
    rotation = rotate.get_90_deg_rotation(doa_list)
    print("rotation: {rotation}".format(rotation=rotation))

    current_position = utility_methods.cylindrical(current_position + rotation)

    iteration += 1

    print("Position at iteration {iteration}: {position}".format(iteration=iteration, position=current_position))
    #   ------------------------------------------------------------------------------------------------------------------
    rotation = rotate.get_45_deg_rotation(doa_list, current_position)

    print("rotation: {rotation}".format(rotation=rotation))

    current_position = utility_methods.cylindrical(current_position + rotation)

    iteration += 1

    print("Position at iteration {iteration}: {position}".format(iteration=iteration, position=current_position))
    #   ------------------------------------------------------------------------------------------------------------------
    rotation = rotate.get_fine_rotation(iteration)

    print("rotation: {rotation}".format(rotation=rotation))

    current_position = utility_methods.cylindrical(current_position + rotation)

    iteration += 1

    print("Position at iteration {iteration}: {position}".format(iteration=iteration, position=current_position))
    #   ------------------------------------------------------------------------------------------------------------------

    rotation = rotate.get_fine_rotation(iteration)

    print("rotation: {rotation}".format(rotation=rotation))

    current_position = utility_methods.cylindrical(current_position + rotation)

    iteration += 1

    print("Position at iteration {iteration}: {position}".format(iteration=iteration, position=current_position))
#   ------------------------------------------------------------------------------------------------------------------
