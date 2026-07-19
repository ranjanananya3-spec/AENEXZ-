def find_largest(a, b, c):
    
    largest = a
    
   
    if b > largest:
        largest = b
        
   
    if c > largest:
        largest = c
        
    return largest


num1 = 45
num2 = 82
num3 = 19

result = find_largest(num1, num2, num3)
print(f"The largest number between {num1}, {num2}, and {num3} is: {result}")
