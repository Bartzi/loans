import tkinter as tk
import json
import os

from PIL import Image, ImageTk


class Rect():
    def __init__(self, canvas, x1, y1, x2, y2):
        self.coords = [x1, y1, x2, y2]
        self.canvas = canvas
        self.make()

    def make(self):
        self.id = self.canvas.create_rectangle(*self.coords, outline="red")

    def update(self):
        self.canvas.coords(self.id, *self.coords)

    def set_br(self, x, y):
        self.coords[2] = x
        self.coords[3] = y
        self.update()

    def finish(self):
        # sort coords
        self.coords = [
            min(self.coords[0], self.coords[2]),
            min(self.coords[1], self.coords[3]),
            max(self.coords[0], self.coords[2]),
            max(self.coords[1], self.coords[3]),
        ]
        self.canvas.itemconfig(self.id, outline="black")

    def delete(self):
        self.canvas.delete(self.id)


class Viewer(tk.Frame):
    def __init__(self, output_folder, master=None, canvas_size=1632, stretch=True, row=0, column=0, rowspan=1, columnspan=1):
        tk.Frame.__init__(self, master)

        self.output_folder = output_folder

        self.canvas_size = (canvas_size, canvas_size // 16 * 10)
        self.stretch = stretch
        self.stretch_factor = 2

        self.rectangles = []

        self.grid(padx=2, pady=2, row=row, column=column, rowspan=rowspan, columnspan=columnspan)
        self.create_canvas()
        self.bind_all("<Right>", self.on_right_pressed)
        self.bind_all("<Left>", self.on_left_pressed)
        self.bind_all("<Button-1>", self.on_mouse_click)
        self.bind_all('<B1-Motion>', self.on_mouse_drag)
        self.bind_all('<ButtonRelease-1>', self.on_mouse_release)

        self.index_var = tk.IntVar()
        self.index_var.set(-1)

        self.path = None

    def make_frame(self, row=0, column=0, rowspan=1, columnspan=1):
        frame = tk.Frame(self, borderwidth=3, relief="ridge")
        frame.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, padx=2, pady=2, sticky=tk.E + tk.W + tk.S + tk.N)
        return frame

    def create_canvas(self):
        self.canvas_frame = self.make_frame(row=3, column=0, columnspan=2)
        self.canvas = tk.Canvas(self.canvas_frame, width=self.canvas_size[0], height=self.canvas_size[1])
        self.canvas.pack(side = tk.BOTTOM)

        self.image_on_canvas = None

    def on_mouse_click(self, event):
        if self.path is None:
            return

        self.rectangles.append(Rect(self.canvas, event.x, event.y, event.x, event.y))

    def on_mouse_drag(self, event):
        if len(self.rectangles) == 0:
            return
        self.rectangles[-1].set_br(event.x, event.y)

    def on_mouse_release(self, event):
        if len(self.rectangles) == 0:
            return
        self.rectangles[-1].finish()

    def on_right_pressed(self, event):
        self.save_current_rectangles()
        self.index_var.set(self.index_var.get() + 1)
        self.on_image_changed()

    def on_left_pressed(self, event):
        self.save_current_rectangles()
        self.index_var.set(self.index_var.get() - 1)
        self.on_image_changed()

    def save_current_rectangles(self):
        if self.path is None:
            return

        os.makedirs(self.output_folder, exist_ok=True)

        base, ext = os.path.splitext(os.path.basename(self.path))
        out_path = os.path.join(self.output_folder, "{}.json".format(base))
        with open(out_path, "w") as out_handle:
            out_handle.write(json.dumps([[y * self.stretch_factor for y in x.coords] for x in self.rectangles]))
        for x in self.rectangles:
            x.delete()
        self.rectangles = []

    def on_image_changed(self, _ = 0):
        if not hasattr(self, 'images'):
            return

        self.update_image(self.index_var.get())

    def update_image(self, index):
        self.index = index

        self.path = self.images[self.index]

        self.img = Image.open(self.path)

        if self.stretch:
            width, height = self.img.size
            self.img = self.img.resize((width // self.stretch_factor, height // self.stretch_factor), Image.ANTIALIAS)

        self.photo = ImageTk.PhotoImage(image = self.img)

        if self.image_on_canvas is None:
            self.image_on_canvas = self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        else:
            self.canvas.itemconfig(self.image_on_canvas, image=self.photo)
