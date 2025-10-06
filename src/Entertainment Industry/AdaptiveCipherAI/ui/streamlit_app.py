import streamlit as st

from ciphers import caesar, substitution, vigenere
from solvers import rule_based


st.set_page_config(page_title="AdaptiveCipherAI", page_icon="🧩")
st.title("AdaptiveCipherAI — Classical Cipher Playground")

mode = st.sidebar.selectbox("Mode", ["Encrypt", "Solve (Rule-based)"])
cipher = st.sidebar.selectbox("Cipher", ["caesar", "substitution", "vigenere"]) 

text = st.text_area("Text", value="HELLO NEW USER")

if mode == "Encrypt":
	if cipher == 'caesar':
		shift = st.number_input("Shift", min_value=0, max_value=25, value=3)
		st.code(caesar.encrypt(text, shift))
	elif cipher == 'substitution':
		key = st.text_input("Key (26-letter perm)", value="QWERTYUIOPASDFGHJKLZXCVBNM")
		try:
			st.code(substitution.encrypt(text, key))
		except Exception as e:
			st.error(str(e))
	else:
		key = st.text_input("Key (letters)", value="LEMON")
		st.code(vigenere.encrypt(text, key))
else:
	if cipher == 'caesar':
		shift, plain = rule_based.solve_caesar(text)
		st.write(f"Shift: {shift}")
		st.code(plain)
	elif cipher == 'substitution':
		key, plain, score = rule_based.solve_substitution(text, max_iter=3000)
		st.write(f"Score: {score:.2f}")
		st.write(f"Key: {key}")
		st.code(plain)
	else:
		st.info("Vigenère solver not implemented in rule-based demo.")


