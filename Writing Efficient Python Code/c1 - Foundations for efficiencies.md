1. ### Welcome.
   
   Hello and welcome! My name is Logan Thomas, and I'll be your guide through this course about writing efficient code using Python.

2. Course overview

    As a data scientist, the majority of your time should be spent gleaning actionable insights from data. Whether you're cleaning and curating a messy dataset, deploying a machine learning model, or creating a sleek data visualization, the code you write should be a helpful tool to quickly get you where you need to go - not something that leaves you waiting around. In this course, you'll learn how to write cleaner, faster, and more efficient Python code. We'll explore how to time and profile your code in order to find potential bottlenecks. Then, you'll practice eliminating these bottlenecks, and other bad design patterns, using Python's Standard Library, NumPy, and pandas. After completing this course, you'll have everything you need to start writing elegant and efficient Python code! But first, let's explore what is meant by efficient Python code.

    - Your code should be a tool used to **gain insights**
      - Not something that leaves you waiting for results
    - In this course, you will learn:
      - How to write clean, fast, and ecient Python code
      - How to prole your code for bolenecks
      - How to eliminate bolenecks and bad design paerns
    

3. Defining efficient
    
    In the context of this course, efficient refers to code that satisfies two key concepts. First, efficient code is fast and has a small latency between execution and returning a result. Second, efficient code allocates resources skillfully and isn't subjected to unnecessary overhead. Although your definition of fast runtime and small memory usage may depend on the task at hand, the goal of writing efficient code is still to reduce both latency and overhead. For the remainder of this course, we'll be exploring how to write Python code that runs quickly and has little memory overhead.

    - Writing ecient Python code
      - Minimal completion time (fast runtime)
      - Minimal resource consumption (small memory footprint

4. Defining Pythonic
   
   We've defined what is meant by efficient code, but it is also important to note that this course focuses on writing efficient code using Python. Python is a language that prides itself on code readability, and thus, it comes with its own set of idioms and best practices. Writing Python code the way it was intended is often referred to as Pythonic code. This means the code that you write follows the best practices and guiding principles of Python. Pythonic code tends to be less verbose and easier to interpret. Although Python supports code that doesn't follow its guiding principles, this type of code tends to run slower. As an example, look at the non-Pythonic code in this slide. Not only is this code more verbose than the Pythonic version, it takes longer to run. We'll take a closer look at why this is the case later on in the course, but for now, the main take away here is that Pythonic code is efficient code!

- Writing ecient Python code
  - Focus on **readability**
  - Using Python's constructs as intended (i.e., Pythonic)


```python
#Non-Pythonic
doubled_numbers = []

for i in range(len(numbers)):
    doubled_numbers.append(numbers[i] * 2)

# Pythonic
doubled_numbers = [x * 2for x in numbers]
```



1. The Zen of Python by Tim Peters

    In the previous slide, I discussed the importance of following Python's guiding principles and idioms. But what are these idioms you may ask? Enter The Zen of Python. This is a list of a few idioms and best practices that summarize Python's design philosophy. Take a moment to familiarize yourself with these principles. Chances are, if you have been working with Python for some time, you are already familiar with most of them. If not, don't worry! We'll be touching on a few of these concepts throughout the course.

- Beautiful is better than ugly.
- Explicit is better than implicit.
- Simple is better than complex.
- Complex is better than complicated.
- Flat is better than nested.
- Sparse is better than dense.
- Readability counts.
- Special cases aren't special enough to break the rules.
- Although practicality beats purity.
- Errors should never pass silently.
- Unless explicitly silenced.
- In the face of ambiguity, refuse the temptation to guess.
- ...

2. Things you should know
    
    Before moving on, we should check your Python knowledge level. There are a few things this course assumes you have a working knowledge of. You don't need to be an expert on the topics listed here, but you should definitely be familiar with them in order to get the most out of this course.

    - Data types typically used in Data Science
      - [Data Types for Data Science](https://learn.datacamp.com/courses/data-types-for-data-science)
    - Writing and using your own functions
      - [Python Data Science Toolbox (Part 1)](https://learn.datacamp.com/courses/python-data-science-toolbox-part-1)
    - Anonymous functions ( lambda expressions)
      - [Python Data Science Toolbox (Part 1)](https://learn.datacamp.com/courses/python-data-science-toolbox-part-1)
    - Writing and using list comprehensions
      - [Python Data Science Toolbox (Part 2)](https://learn.datacamp.com/courses/python-data-science-toolbox-part-2)

3. Let's get started!
    
    Now that we've defined efficient and Pythonic code, and touched on a few things you should already be familiar with, it's time to start coding!



------------

# Building with builtins

## The Python Standard Library

- Python 3.6 Standard Library
  - Part of every standard Python installation
- Built-in types
  - `list` , `tuple` , `set` , `dict` , and others
- Built-in functions
  - `print()` , `len()` , `range()` , `round()` , `enumerate()` , `map()` , `zip()` , and others
- Built-in modules
  - `os` , `sys` , `itertools` , `collections` , `math` , and others


### Built-in function: range()

Explicitly typing a list of numbers
`nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`

### Using range() to create the same list


```python
# range(start,stop)
nums = range(0,11)

nums_list = list(nums)
print(nums_list)

[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```


```python
# range(stop)
nums = range(11)

nums_list = list(nums)
print(nums_list)

> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

### Built-in function: range()

Using `range()` with a step value


```python
even_nums = range(2, 11, 2)

even_nums_list = list(even_nums)
print(even_nums_list)

> [2, 4, 6, 8, 10]
```

### Built-in function: enumerate()

Creates an indexed list of objects

```python
letters = ['a','b','c','d']

indexed_letters = enumerate(letters)

indexed_letters_list = list(indexed_letters)
print(indexed_letters_list)

> [(0,'a'), (1,'b'), (2,'c'), (3,'d')]
```

Can specify a start value

```python
letters = ['a','b','c','d']

indexed_letters2 = enumerate(letters, start=5)

indexed_letters2_list = list(indexed_letters2)
print(indexed_letters2_list)

> [(5,'a'), (6,'b'), (7,'c'), (8,'d')]ç
```
### Built-in function: map()

Applies a function over an object

```python
nums = [1.5, 2.3, 3.4, 4.6, 5.0]

rnd_nums = map(round, nums)

print(list(rnd_nums))

> [2, 2, 3, 5, 5]
```

`map()` with `lambda` (anonymous function)


```python
nums = [1, 2, 3, 4, 5]

sqrd_nums = map(lambda x: x ** 2, nums)

print(list(sqrd_nums))

> [1, 4, 9, 16, 25]
```

# The power of NumPy arrays

## NumPy array overview

Alternative to Python lists


```python
nums_list = list(range(5))
> [0, 1, 2, 3, 4]
```



```python
import numpy as np
nums_np = np.array(range(5))

> array([0, 1, 2, 3, 4])
```

```python
# NumPy array homogeneity
nums_np_ints = np.array([1, 2, 3])
> array([1, 2, 3])

nums_np_ints.dtype
> dtype('int64')

nums_np_floats = np.array([1, 2.5, 3])
> array([1. , 2.5, 3. ])

nums_np_floats.dtype
> dtype('float64')
```

### NumPy array broadcasting

- Python lists don't support broadcasting

```python
nums = [-2,-1, 0, 1, 2]
nums ** 2
> TypeError: unsupported operand type(s) for ** or pow(): 'list' and 'int'
```

- List approach

```python
# For loop (inefficient option)
sqrd_nums = []

for num in nums:
    sqrd_nums.append(num ** 2)
    print(sqrd_nums)

> [4, 1, 0, 1, 4]
```

```python
# List comprehension (better option but not best)
sqrd_nums = [num ** 2 for num in nums]

print(sqrd_nums)

> [4, 1, 0, 1, 4]
```
- NumPy array broadcasting for the win!

```python
nums_np = np.array([-2,-1, 0, 1, 2])
nums_np ** 2

> array([4, 1, 0, 1, 4])
```
