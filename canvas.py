import csv
import tkinter as tk

# Create a canvas for drawing
root = tk.Tk()
canvas = tk.Canvas(root, width=500, height=500, bg="white")
canvas.pack()

# Initialize the list of points
points = []


# Define the function to save the points to a CSV file
def save_points():
    with open('points.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'color'])
        for point in points:
            res = 0
            if point[2] == 'blue':
                res = 1
            writer.writerow([point[0], point[1], res])


# Define the function to draw a point on the canvas
def draw_point(event):
    x, y = event.x, event.y

    # Get the current color
    color = current_color.get()

    # Add the point to the list of points
    points.append((x, y, color))

    # Draw the point on the canvas
    canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color)


# Define the function to switch the current color
def switch_color():
    global current_color
    if current_color.get() == 'blue':
        current_color.set('orange')
    else:
        current_color.set('blue')


# Create a StringVar to store the current color
current_color = tk.StringVar()
current_color.set('blue')

# Create a button to switch the color
color_button = tk.Button(root, text="Switch Color", command=switch_color)
color_button.pack()

# Bind the mouse click event to the draw_point function
canvas.bind('<ButtonPress-1>', draw_point)

# Bind the save button to the save_points function
save_button = tk.Button(root, text="Save", command=save_points)
save_button.pack()

# Start the tkinter event loop
root.mainloop()
