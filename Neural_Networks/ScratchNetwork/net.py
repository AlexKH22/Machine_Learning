from tkinter import *
import netinit
from tkinter.filedialog import askopenfilename


class Window(object):
    """
        GUI

        Attributes:
        menubar: Menu;
        filemenu: 1-st list of menu;
        helpemenu: 2-st list of menu;

        fields: Labels in GUI;
        ents: Input objects;
        separator: Separator between elements;

        file_but: Button responsible for file choice;
        train_but: Button responsible for model training;

        train_widget: TextBox;
        scroll: Scroll on TextBox.
    """
    def __init__(self, app):
        # Set size and title of the window
        app.geometry('300x450+500+300')
        app.title("Animals")

        # Menu
        self.menubar = Menu(app)

        self.filemenu = Menu(self.menubar)
        self.menubar.add_cascade(label="Animals", menu=self.filemenu)
        self.filemenu.add_command(label="Exit", command=app.quit)

        self.helpmenu = Menu(self.menubar)
        self.menubar.add_cascade(label="Help", menu=self.helpmenu)
        self.helpmenu.add_command(label="About", command=self.about)

        app.config(menu=self.menubar)

        # Fields for labels
        self.fields = ('Learning Rate', 'Number of Iterations')

        # Input objects
        self.ents = self.makeform(app)

        # Separator between elements
        self.separator = Frame(height=2, bd=1, relief=SUNKEN)
        self.separator.pack(fill=X, pady=5)

        # Buttons
        self.file_but = Button(self.separator, text='Choose file',
                               command=self.choose_file)
        self.file_but.pack(side=TOP)

        self.train_but = Button(self.separator, text='Train Model',
                                command=(lambda e=self.ents: self.train_model(e)))
        self.train_but.pack(side=TOP)

        # Separator between elements
        self.separator = Frame(height=2, bd=1, relief=SUNKEN)
        self.separator.pack(fill=X, pady=5)

        # TextBox scroll
        self.text_widget = Text(app, height=20, width=39)
        self.scroll = Scrollbar(app, command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=self.scroll.set)
        self.scroll.pack(side=RIGHT, fill=Y)

        # Special formats for TextBox
        self.text_widget.tag_configure('big', font=('Verdana', 18, 'bold'), justify=CENTER)
        self.text_widget.tag_configure('color', font=('Verdana', 8))
        self.text_widget.tag_configure('author', font=('Verdana', 16, 'bold'), justify=CENTER)
        self.text_widget.tag_configure('warning', font=('Verdana', 10, 'bold'))
        self.text_widget.pack(side=LEFT, padx=2)

        # Call method about
        self.about()

    """
    The method which opens file dialog
    """
    def choose_file(self):
        self.text_widget.delete(1.0, 'end')
        file_path = askopenfilename()
        # If valid file
        if 'zoo.data.txt' in file_path:
            netinit.data_creation(file_path)
            self.text_widget.insert(END, 'File is downloaded\n', 'big')
            self.file_but.config(state=DISABLED)
        else:
            self.text_widget.insert(END, 'Choose a file with\n'
                                    'name: zoo.data.txt\n', 'big')

    """
    The method which trains model

    Parameters:
    entries: Input objects.
    """
    def train_model(self, entries):
        netinit.train(entries, self.text_widget)

    """
    The method which creates Labels and TextInputs

    Parameters:
    interface: Main window.
    """
    def makeform(self, interface):
        # Create a dictionary with Input Objects
        entries = {}
        # For every Input Object
        for field in self.fields:
            # Frame for Label and Input
            row = Frame(interface)
            lab = Label(row, width=20, text=field + ": ", anchor='w')
            ent = Entry(row, width=10)
            if field == 'Number of Iterations':
                ent.insert(0, "100")
            else:
                ent.insert(0, "0.01")
            row.pack(side=TOP, fill=X, padx=5, pady=5)
            lab.pack(side=LEFT)
            ent.pack(side=LEFT)
            entries[field] = ent
        return entries

    """
    The method which fills the TextBox with information about
    """
    def about(self):
        self.text_widget.delete(1.0, 'end')
        self.text_widget.insert(END, 'About', 'big')
        self.text_widget.insert(END, '\nThis is a 3 layer Neural Network which is able\n'
                                'to predict a type of the specific animal after'
                                '\nthe proper training.\n'
                                '30 neurons in hidden layer\n\n'
                                'Optimal values:\n'
                                'Learning rate(0<x<0.1): 0.01;\n'
                                'Iterations(9<x<inf): 40-10000.\n\n'
                                'It is possible to retrain a model a lot of times\n\n'
                                "Results will be located in 'result.txt'\n\n\n\n\n", 'color')
        self.text_widget.insert(END, 'Alexander Kharchistov 2016', 'author')

if __name__ == '__main__':
    app = Tk()

    window = Window(app)

    app.mainloop()
