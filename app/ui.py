import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from threading import Thread
from app.ranking import get_rankings, download

# Overall scaling factor
SCALE = 1.8


def scaled(value):
    return int(value * SCALE)


class AlternatingListbox(tk.Listbox):
    def __init__(self, master=None, **kwargs):
        tk.Listbox.__init__(self, master, **kwargs)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Return>", self._on_enter)  # Add binding for Enter key
        self.bind("<<ListboxSelect>>", self._on_select)
        self.configure(activestyle="none")  # Remove default selection highlight
        self.configure(
            selectbackground="#e0e0e0"
        )  # Set selection background to match alternating color

    def _on_click(self, event):
        self.selection_clear(0, tk.END)
        index = self.nearest(event.y)
        self.selection_set(index)
        self.activate(index)
        self._recolor()

    def _on_select(self, event):
        self._recolor()

    def _on_enter(self, event):
        self.event_generate("<<ItemActivate>>")  # Generate a custom event

    def _recolor(self):
        for i in range(self.size()):
            if i % 2 == 0:
                self.itemconfigure(i, background="#f0f0f0")  # Light gray
            else:
                self.itemconfigure(i, background="#e0e0e0")  # Slightly darker gray

    def insert(self, index, *elements):
        super().insert(index, *elements)
        self._recolor()


def create_main_window():
    root = tk.Tk()
    root.title("Text Processing App")
    root.geometry(f"{scaled(400)}x{scaled(450)}")

    default_font = ("TkDefaultFont", scaled(10))

    label = tk.Label(root, text="Enter your query:", font=default_font)
    label.pack(pady=scaled(10))

    entry = tk.Entry(root, width=scaled(50), font=default_font)
    entry.pack(pady=scaled(10))

    submit_button = tk.Button(
        root,
        text="Submit",
        command=lambda: on_submit(entry, result_listbox, root),
        font=default_font,
    )
    submit_button.pack(pady=scaled(10))

    # Create a frame to hold the listbox and scrollbar
    list_frame = tk.Frame(root)
    list_frame.pack(pady=scaled(10), fill=tk.BOTH, expand=True)

    # Create the scrollbar
    scrollbar = tk.Scrollbar(list_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    result_listbox = AlternatingListbox(
        list_frame,
        height=scaled(10),
        width=scaled(50),
        font=default_font,
        yscrollcommand=scrollbar.set,
    )
    result_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar.config(command=result_listbox.yview)
    result_listbox.bind("<Double-Button-1>", lambda event: on_item_click(event, root))
    result_listbox.bind("<<ItemActivate>>", lambda event: on_item_activate(event, root))

    # root.result_data = {}  # Store result data in the root widget
    root.result_data = []  # Store result data as a list in the root widget
    return root


def on_item_click(event, root):
    widget = event.widget
    index = widget.nearest(event.y)
    if 0 <= index < len(root.result_data):
        title, description = root.result_data[index]
        show_description(root, title, description)
    else:
        messagebox.showinfo("Error", "Description not found")


def on_item_activate(event, root):
    widget = event.widget
    selection = widget.curselection()
    if selection:
        index = selection[0]
        if 0 <= index < len(root.result_data):
            title, description = root.result_data[index]
            show_description(root, title, description)
        else:
            messagebox.showinfo("Error", "Description not found")


def show_description(root, title, description):
    popup = tk.Toplevel(root)
    popup.title(title)
    popup.geometry(f"{scaled(400)}x{scaled(300)}")

    button_frame = tk.Frame(popup)
    button_frame.pack(fill="x", padx=scaled(10), pady=(scaled(10), 0))

    # Function to close the popup
    def close_popup(event=None):
        popup.destroy()

    def download_content():
        try:
            download(title)
            messagebox.showinfo("Download", "Content downloaded successfully!")
        except Exception as e:
            messagebox.showerror(
                "Download Error", f"An error occurred during download: {str(e)}"
            )

    download_button = tk.Button(
        button_frame,
        text="Download",
        command=download_content,
        font=("TkDefaultFont", scaled(10), "bold"),
    )
    download_button.pack(side="left")

    text = tk.Text(popup, wrap=tk.WORD, font=("TkDefaultFont", scaled(10)))
    text.pack(expand=True, fill="both", padx=scaled(10), pady=scaled(10))
    text.insert(tk.END, description)
    text.config(state="disabled")

    # Bind Esc key to close_popup function
    popup.bind("<Escape>", close_popup)
    # Make sure the popup window takes focus
    popup.focus_set()


def on_submit(entry, result_listbox, root):
    def run_task():
        input_text = entry.get()

        if input_text:
            result = get_rankings(input_text)

            def update_ui():
                result_listbox.delete(0, tk.END)
                root.result_data.clear()

                if not result:
                    result_listbox.insert(tk.END, "No results from processing")
                    return

                for i, item in enumerate(result):
                    if isinstance(item, tuple) and len(item) == 2:
                        title, description = item
                    else:
                        raise ValueError(item)

                    result_listbox.insert(tk.END, title)
                    if i % 2 == 0:
                        result_listbox.itemconfigure(
                            i, background="#f0f0f0"
                        )  # Light gray
                    else:
                        result_listbox.itemconfigure(
                            i, background="#e0e0e0"
                        )  # Slightly darker gray
                    root.result_data.append((title, description))

            root.after(0, update_ui)
        else:
            root.after(0, lambda: result_listbox.delete(0, tk.END))
            root.after(
                0, lambda: result_listbox.insert(tk.END, "Please enter some text.")
            )

    thread = Thread(target=run_task)
    thread.start()


if __name__ == "__main__":
    root = create_main_window()
    root.mainloop()
