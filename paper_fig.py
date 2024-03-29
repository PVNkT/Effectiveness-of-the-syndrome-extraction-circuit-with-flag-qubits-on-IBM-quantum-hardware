
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from paper_utils import paper_utils


def get_option():

    operation = selected_operation.get()
    draw_class = paper_utils()
    draw_function = getattr(draw_class, f'draw_{operation}')

    fig = draw_function()

    if hasattr(root, 'canvas'):
        # Update existing canvas
        root.canvas.get_tk_widget().destroy()

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(in_=graph_frame, side='right')
    root.canvas = canvas

# Create the main window
root = tk.Tk()
root.title("Drawer")
input_frame = tk.Frame(root)
input_frame.pack(side="left")

# Create and pack radiobuttons for operations
selected_operation = tk.StringVar()

option = tk.Radiobutton(input_frame, text="Fig.8 (b)", variable=selected_operation, value='ECDF_8')
option.pack()

#option = tk.Radiobutton(input_frame, text="Table 1", variable=selected_operation, value='hardware_weight')
#option.pack()

option = tk.Radiobutton(input_frame, text="Fig.9", variable=selected_operation, value='syndrome_graph_9')
option.pack()

#option = tk.Radiobutton(input_frame, text="Table 2", variable=selected_operation, value='Sample_weight')
#option.pack()

option = tk.Radiobutton(input_frame, text="Fig.10 (a)", variable=selected_operation, value='Correlation_matrix_0')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.10 (b)", variable=selected_operation, value='Correlation_matrix_1')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.10 (c)", variable=selected_operation, value='Correlation_matrix_2')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.11 (a)", variable=selected_operation, value='defect_syndrome_0')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.11 (b)", variable=selected_operation, value='defect_syndrome_1')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.11 (c)", variable=selected_operation, value='defect_syndrome_2')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.12 (a)", variable=selected_operation, value='Logical_error_rate_hardware_0')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.12 (b)", variable=selected_operation, value='Logical_error_rate_hardware_1')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.12 (c)", variable=selected_operation, value='Logical_error_rate_hardware_2')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.12 (d)", variable=selected_operation, value='Logical_error_rate_sample_0')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.12 (e)", variable=selected_operation, value='Logical_error_rate_sample_1')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.12 (f)", variable=selected_operation, value='Logical_error_rate_sample_2')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.17 (a)", variable=selected_operation, value='ECDF_3')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.17 (b)", variable=selected_operation, value='ECDF_5')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.17 (c)", variable=selected_operation, value='ECDF_7')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.17 (d)", variable=selected_operation, value='ECDF_9')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.20 (a)", variable=selected_operation, value='Correlation_matrix_3')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.20 (b)", variable=selected_operation, value='Correlation_matrix_5')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.20 (c)", variable=selected_operation, value='Correlation_matrix_7')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.20 (d)", variable=selected_operation, value='Correlation_matrix_9')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.21 (a)", variable=selected_operation, value='Correlation_matrix_invert_3')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.21 (b)", variable=selected_operation, value='Correlation_matrix_invert_5')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.21 (c)", variable=selected_operation, value='Correlation_matrix_invert_7')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.21 (d)", variable=selected_operation, value='Correlation_matrix_invert_9')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.22 (a)", variable=selected_operation, value='defect_probability_real')
option.pack()

#option = tk.Radiobutton(input_frame, text="Fig.22 (b)", variable=selected_operation, value='defect_probability_stim')
#option.pack()

option = tk.Radiobutton(input_frame, text="Fig.23 (a)", variable=selected_operation, value='Logical_error_rate_hardware_Z1_log')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.23 (b)", variable=selected_operation, value='Logical_error_rate_hardware_Z1_linear')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.23 (c)", variable=selected_operation, value='Logical_error_rate_hardware_X1_log')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.23 (d)", variable=selected_operation, value='Logical_error_rate_hardware_X1_linear')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.24 (a)", variable=selected_operation, value='Sample_probability_hardware')
option.pack()

option = tk.Radiobutton(input_frame, text="Fig.24 (b)", variable=selected_operation, value='Sample_probability_sample')
option.pack()
# Button to calculate and plot the result
calculate_button = tk.Button(root, text="Draw", command=get_option)
calculate_button.pack(side='left')

# Label to display the result
result_label = tk.Label(root, text="")
result_label.pack()

# Frame to contain the graph
graph_frame = tk.Frame(root)
graph_frame.pack(side='right')

# Start the Tkinter event loop
root.mainloop()
