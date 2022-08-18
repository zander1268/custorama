import streamlit as st
from PIL import Image



#Header
st.image("Images/penguins.jpg", use_column_width="always")
st.title("Custorama CLV Predictions")
st.header("This is a header")
#Columns
col11, col12 = st.columns(2)
col11.subheader("Column 1")
col12.subheader("Column 2")

col21, col22, col23 = st.columns([3,2,1])
col21.write("Large column Text will wrap around if there is enough space")
col22.write("medium column")
col23.write("small column")
#Markdown
st.markdown("Markdown **syntax** *works*")
'Markdown'
'## Magic'
st.write('<h2 style="text-align:center">Text aligned with Html</h2>',)
#Widgets
st.header("Widgets")
button1 = st.button("This is a button")
if button1:
    st.write("You clicked button")
check = st.checkbox("Please check this box")
if check:
    st.write("You checked box")
else:
    st.write("The box was not checked")
#Radio button
st.subheader("Radio Button")
animal_options = ["Cats","Dogs","Pigs"]
fav_animal = st.radio("Which animal is your favorite?",animal_options)
button2 = st.button("Submit animal")
if button2:
    st.write(f'You selected {fav_animal} as your fav animal')
#Multi select
like_animals = st.multiselect("Which animals do you like?",animal_options)
st.write(like_animals)
st.write(f'The animal you liked first was {like_animals[0]}')
#slider
num_pets = st.slider("How many pets is too many?",2,20,2)
#Text input
pet_name = st.text_input("What is your pets name?",value="I don't have a pet")
st.write(pet_name)

#Sidebar
st.sidebar.title("Sidebar")
side_button = st.sidebar.button("Press Me")
if side_button:
    st.write("Button was pressed")



