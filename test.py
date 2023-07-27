import random

def test_random_choices_with_weights():
    options = ['Option 1', 'Option 2', 'Option 3']
    weights = [1,1,1]  # The weights for each option

    for _ in range(10):
        # Make 1000 selections using random.choices with the specified weights
        selections = random.choices(options, weights=weights, k=1)
        print(selections)

if __name__ == "__main__":
    test_random_choices_with_weights()
