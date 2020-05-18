import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def convert_column_name_to_label(x):

    if x == 'KIC':
        return r'KIC'
    if x == 'numax':
        return r'$\nu_{\mathrm{max}}$ ($\mu$Hz)'
    if x == 'var':
        return r'MAD'
    elif x == 'zc':
        return 'Normalised zero-crossings'
    elif x == 'hoc':
        return r'Coherency $\psi^{2}$'
    elif x == 'abs_k_mag':
        return r'$M_{\mathrm{K}_{s}}$'

st.write("""
# Clumpiness Data Explorer!
Welcome!
""")

@st.cache
def load_data():
    df = pd.read_csv("Colours_New_Gaps_output_data_noise_KIC_-1_APOKASC.csv")
    return (df)

df = load_data()
st.sidebar.title('Plotting Options')

option_x = st.sidebar.selectbox(
        'Which x-variable do you like best?',
        df.columns)
st.sidebar.markdown("You selected: "+str(convert_column_name_to_label(option_x)))

option_y = st.sidebar.selectbox(
        'Which y-variable do you like best?',
        df.columns)
st.sidebar.markdown("You selected: "+str(convert_column_name_to_label(option_y)))

x_min, x_max = st.slider(option_x, 
                         df[option_x].min(),
                         df[option_x].max(),
                        [df[option_x].min(),
                         df[option_x].max()],
                         key=0)
y_min, y_max = st.slider(option_y, 
                         df[option_y].min(),
                         df[option_y].max(),
                        [df[option_y].min(),
                         df[option_y].max()],
                        key=1)
print(y_min, y_max)
plt.scatter(df.loc[(df[option_x] > x_min) &
                   (df[option_x] < x_max) &
                   (df[option_y] > y_min) &
                   (df[option_y] < y_max), option_x], 
            df.loc[(df[option_x] > x_min) &
                   (df[option_x] < x_max) &
                   (df[option_y] > y_min) &
                   (df[option_y] < y_max), option_y], 
            s=3)
plt.xlabel(convert_column_name_to_label(option_x))
plt.ylabel(convert_column_name_to_label(option_y))
plt.xlim(x_min*0.9, x_max*1.1)
plt.ylim(y_min*0.9, y_max*1.1)
if st.sidebar.checkbox('Logarithmic x-variable'):
    plt.xscale('log')
if st.sidebar.checkbox('Logarithmic y-variable'):
    plt.yscale('log')

st.pyplot()