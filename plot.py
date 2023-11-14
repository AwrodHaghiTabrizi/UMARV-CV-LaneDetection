import matplotlib.pyplot as plt




file_path = 'comp23_4_3.txt'  # Replace with the path to your text file
try:
    with open(file_path, 'r') as file:
        # Step 2: Read the contents of the file
        file_content = file.read()

        # Step 3: Split the content into individual numbers (assuming space-separated)
        numbers_as_strings = file_content.split()

        # Step 4: Convert the strings to numbers (e.g., integers or floats)
        numbers = [float(num) for num in numbers_as_strings]  # Use int() for integers or float() for floats

        # Step 5: Store the numbers in a list
        #print(numbers)  # This list contains your numbers
except FileNotFoundError:
    print(f"The file '{file_path}' does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")

#print(numbers)
#count = 0
x = []
y = []
numbers.pop(0)
for count, i in enumerate(numbers):
    if(count % 2 == 0):
        x.append(i * 640)
    else:
        y.append(i * 384)

print(x)

# plt.scatter(x, y, label= "stars", color= "green",  
#     marker= "*", s=30) 

# # x-axis label 
# plt.xlabel('x - axis') 
# # frequency label 
# plt.ylabel('y - axis') 
# # plot title 
# plt.title('My scatter plot!') 
# # showing legend 
# plt.legend() 
# plt.show()

plt.plot(x, y, label = "line 2")

plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 
# giving a title to my graph 
plt.title('Two lines on same graph!') 
  
# show a legend on the plot 
plt.legend() 
  
# function to show the plot 
plt.show()