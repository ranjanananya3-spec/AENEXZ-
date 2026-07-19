#question1
import math

def circle_area(radius):
    """Return the area of a circle given its radius."""
    return math.pi * radius * radius

#question2
def largest_of_three(a, b, c):
    """Return the largest of three numbers."""
    return max(a, b, c)

#question3
def string_length(text):
    """Return the length of a string without using built-in length functions."""
    count = 0
    for char in text:
        count += 1
    return count

#question4
def is_prime(number):
    """Return True if a number is prime, otherwise False."""
    if number <= 1:
        return False
    if number == 2:
        return True
    if number % 2 == 0:
        return False

    divisor = 3
    while divisor * divisor <= number:
        if number % divisor == 0:
            return False
        divisor += 2

    return True

#question5
def fibonacci(n):
    """Return the nth Fibonacci number using recursion."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

#question6
def reverse_string(text):
    """Return the reverse of a string using recursion."""
    if len(text) == 0:
        return ""
    return reverse_string(text[1:]) + text[0]

#question7
def sum_upto_n(n):
    """Return the sum of all numbers from 1 to n using a for loop."""
    total = 0
    for i in range(1, n + 1):
        total += i
    return total

#question8
def count_even_numbers(numbers):
    """Return the number of even values in a list."""
    count = 0
    for num in numbers:
        if num % 2 == 0:
            count += 1
    return count

#question9
for number in range(2, 101):
    if is_prime(number):
        print(number)

