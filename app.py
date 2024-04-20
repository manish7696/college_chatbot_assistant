# from tkinter import *
# from chat import get_response, bot_name

# background_color = "#17202A"
# text_color = "#EAECEE"

# class chatApplication:
#     def __init__(self):
#         self.window = Tk()
#         self.setup()

#     def run(self):
#         self.window.mainloop()

#     def setup(self):
#         self.window.title("DAV Helper")
#         self.window.configure(width=500, height=700, bg=background_color)

#         # Set up the chat display area (70% of the screen height)
#         self.chat_frame = Frame(self.window, bg="#1C4E57")
#         self.chat_frame.place(relwidth=1, relheight=0.85)

#         # Text widget for chat display
#         self.text_widget = Text(self.window, bg="#111223", fg=text_color, state=DISABLED, wrap=WORD)
#         self.text_widget.place(relwidth=0.85, relheight=0.1, rely=0.875)

#         # send button 
#         self.send_button = Button(self.window, text="Send", bg="#ABB2B9", fg="#000000")
#         self.send_button.place(relwidth=0.15, relheight=0.1, rely=0.875, relx=0.85)
        
#         # scroll wala option
#         self.scrollbar = Scrollbar(self.chat_frame, command=self.text_widget.yview)
#         self.text_widget.config(yscrollcommand=self.scrollbar.set)
#         self.scrollbar.place(relx=0.975, rely=0, relwidth=0.025, relheight=1)


# if __name__ == "__main__":
#     app = chatApplication()
#     app.run()


from tkinter import *
from chat import get_response, bot_name

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

class ChatApplication:
    
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()
        
    def run(self):
        self.window.mainloop()
        
    def _setup_main_window(self):
        self.window.title("Chat")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=BG_COLOR)
        
        # head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="Welcome", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)
        
        # tiny divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)
        
        # text widget
        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
                                font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)
        
        # scroll bar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)
        
        # bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)
        
        # message entry box
        self.msg_entry = Entry(bottom_label, bg="#2C3E50", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        
        # send button
        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=BG_GRAY,
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)
     
    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")
        
    def _insert_message(self, msg, sender):
        if not msg:
            return
        
        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)
        
        msg2 = f"{bot_name}: {get_response(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)
        
        self.text_widget.see(END)
             
        
if __name__ == "__main__":
    app = ChatApplication()
    app.run()