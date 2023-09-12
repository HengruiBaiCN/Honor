def create_pyramid(input_list):
    n = len(input_list)
    layers = 0
    while n > 0:
        layers += 1
        n -= layers

    if n < 0:
        raise ValueError("Input list cannot be transformed into a pyramid.")

    pyramid = []
    current_index = 0

    for layer in range(1, layers + 1):
        row = input_list[current_index:current_index + layer]
        pyramid.append(row)
        current_index += layer

    return pyramid

# Example usage:
input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
pyramid = create_pyramid(input_list)
for row in pyramid:
    print(row)