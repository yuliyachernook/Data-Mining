numbers = [1, 2, 3, 4, 5]
numbers2 = list(numbers)
print(numbers[0])
print(numbers[-1])
numbers[0] = 1555
print(numbers[0])
numbers=[5]*6
print(numbers)
for item in numbers:
    print(item)
if numbers == numbers2:
    print("numbers equal to numbers2")
else:
    print("numbers is not equal to numbers2")
numbers.insert(1, "ins")
numbers.append("app")
numbers.remove(numbers[-1])
numbers.clear()
