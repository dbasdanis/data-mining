from tkinter import *
import tkinter as tk
import traceback
import weka.core.jvm as jvm
import weka.core.serialization as serialization
from weka.core.converters import Loader
from weka.classifiers import Classifier
from weka.core.dataset import Instance
from weka.core.dataset import Instances
from weka.core.dataset import Attribute
from PIL import Image, ImageTk


class MainApplication(tk.Frame):
	def __init__(self, parent, *args, **kwargs):
		tk.Frame.__init__(self, parent, *args, **kwargs)
		
		container = tk.Frame(parent)
		container.pack(fill="both", expand=True)
		container.grid_rowconfigure(0, weight=1)
		container.grid_columnconfigure(0, weight=1)
 
		self.background = ImageTk.PhotoImage(Image.open("background.jpg"))

		self.frames = {}
		for F in (Home, Attributes, Result):
			page_name = F.__name__
			frame = F(parent=container, controller=self)
			self.frames[page_name] = frame

			frame.grid(row=0, column=0, sticky="nsew")

		self.show_frame("Home")
		
	def show_frame(self, page_name):
		'''Show a frame for the given page name'''
		frame = self.frames[page_name]
		frame.tkraise()
		
		return frame
		
class Home(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		
		self.controller = controller
		self.background = tk.Label(self,image = self.controller.background)
		self.background.place(x=0, y=0, relwidth=1, relheight=1)
		
		self.grid_rowconfigure(0,weight=1)
		self.grid_rowconfigure(3,weight=1)
		self.grid_columnconfigure(0,weight=1)

		self.question = tk.Label(self, text = "No more gossip...", font = ("Times", 45), bg="DarkGoldenrod1", fg="black")
		self.question.grid(row=1, column=0)
		
		self.answer = tk.Label(self, text = "You have a crush on someone? Find out if he's free!", font = ("Times", 30), bg="DarkGoldenrod1", fg="white")
		self.answer.grid(row=2, column=0)
		
		self.button = tk.Button(self, text = "Continue", font = ("Times", 20), bg="DarkGoldenrod1", fg="black", command=lambda: controller.show_frame("Attributes"))
		self.button.grid(row=4, column=0)
	
class Attributes(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		
		self.controller = controller
		self.background = tk.Label(self,image = self.controller.background)
		self.background.place(x=0, y=0, relwidth=1, relheight=1)
		
		self.grid_rowconfigure(0,weight=1)
		self.grid_rowconfigure(7,weight=1)
		self.grid_columnconfigure(0,weight=1)
		self.grid_columnconfigure(3,weight=1)

		self.gender_label = tk.Label(self, text = "Gender", font = ("Times", 30), bg="DarkGoldenrod1", fg="black")
		self.gender_label.grid(row=1, column=1, sticky = E)
		
		self.gender_entry = Entry(self,font=('Times',30),justify="center",width=10)
		self.gender_entry.grid(row=1, column=2)
		
		
		self.age_label = tk.Label(self, text = "Age", font = ("Times", 30), bg="DarkGoldenrod1", fg="white")
		self.age_label.grid(row=2, column=1, sticky = E)
		
		self.age_entry = Entry(self,font=('Times',30),justify="center",width=10,textvariable='int')
		self.age_entry.grid(row=2, column=2)
		
		
		self.height_label = tk.Label(self, text = "Height", font = ("Times", 30), bg="DarkGoldenrod1", fg="black")
		self.height_label.grid(row=3, column=1, sticky = E)
		
		self.height_entry = Entry(self,font=('Times',30),justify="center",width=10)
		self.height_entry.grid(row=3, column=2)
		
		
		self.weight_label = tk.Label(self, text = "Weight", font = ("Times", 30), bg="DarkGoldenrod1", fg="white")
		self.weight_label.grid(row=4, column=1, sticky = E)
		
		self.weight_entry = Entry(self,font=('Times',30),justify="center",width=10)
		self.weight_entry.grid(row=4, column=2)
		
		
		self.sociability_label = tk.Label(self, text = "Sociability", font = ("Times", 30), bg="DarkGoldenrod1", fg="black")
		self.sociability_label.grid(row=5, column=1, sticky = E)
		
		self.sociability_entry = Entry(self,font=('Times',30),justify="center",width=10)
		self.sociability_entry.grid(row=5, column=2)
		
		
		self.stability_label = tk.Label(self, text = "Stability", font = ("Times", 30), bg="DarkGoldenrod1", fg="white")
		self.stability_label.grid(row=6, column=1, sticky = E)
		
		self.stability_entry = Entry(self,font=('Times',30),justify="center",width=10)
		self.stability_entry.grid(row=6, column=2)
		
		
		self.button = tk.Button(self, text="Predict", font = ("Times", 20), bg="DarkGoldenrod1", fg="black", command=lambda: self.predBtn_clicked())
		self.button.grid(row=8, column=1, columnspan=2)
		
	def clear(self):
		self.gender_entry.delete(0, 'end')
		self.age_entry.delete(0, 'end')
		self.height_entry.delete(0, 'end')
		self.weight_entry.delete(0, 'end')
		self.sociability_entry.delete(0, 'end')
		self.stability_entry.delete(0, 'end')
		
	def BMI(self, weight, height):
		return weight/(height/100)**2
		
	def predBtn_clicked(self):	
		
		gender = self.gender_entry.get()
		age = int(self.age_entry.get())
		height = int(self.height_entry.get())
		weight = int(self.weight_entry.get())
		sociability = self.sociability_entry.get()
		stability = self.stability_entry.get()
		
		'''Create the model'''
		objects = serialization.read_all("J48.model")
		
		cls = Classifier(jobject=objects[0])
		data = Instances(jobject=objects[1])
		
		'''Create the test set to be classified'''
		gender_values = ["Man", "Woman"]
		sociability_values = ["Introvert", "Extrovert"]
		stability_values = ["Stable", "Unstable"]
		
		values = [gender_values.index(gender), age, height, weight, self.BMI(weight, height), stability_values.index(stability), sociability_values.index(sociability), Instance.missing_value()]
		
		inst = Instance.create_instance(values)
		inst.dataset = data
		
		'''Classification'''
		prediction = int(cls.classify_instance(inst))
		self.controller.show_frame("Result").show(prediction)
		self.clear()


class Result(tk.Frame):

	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		self.controller = controller

		self.background = tk.Label(self,image = self.controller.background)
		self.background.place(x=0, y=0, relwidth=1, relheight=1)
		
		self.grid_rowconfigure(0,weight=1)
		self.grid_rowconfigure(3,weight=1)
		self.grid_columnconfigure(0,weight=1)
		
		self.resultTitle = tk.Label(self, text = "It's not private anymore!", font=("Times", 30), bg="DarkGoldenrod1", fg="black")
		self.resultTitle.grid(row=1, column=0)
		
		self.result = tk.Label(self)	
		self.result.grid(row=2, column=0)
		
		self.returnBtn = tk.Button(self, text = "Return", font = ("Times", 20), bg="DarkGoldenrod1", fg="black", command=lambda: controller.show_frame("Home"))
		self.returnBtn.grid(row=4,column=0)
		
	def show(self, prediction):
		classes = ["In Relationship", "Single"]
		self.result.config(text=str(classes[prediction]), font=("Times", 35), bg="DarkGoldenrod1", fg="white")
		
def main():	
		root = Tk()
		root.title("Find out who's in relationship!")
		root.state('zoomed')
		app = MainApplication(root)
		root.mainloop()		
		
if __name__ == "__main__":

	try:
		jvm.start()
		main()
		jvm.stop()
		
	except Exception:
		print(traceback.format_exc())
		
	finally:
		jvm.stop()