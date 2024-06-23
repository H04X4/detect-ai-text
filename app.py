from googletrans import Translator
import streamlit as st

from detect.detect_ai import *
from models.llama import llama_gen
from models.gemma import gemma_gen
from models.qwen2 import qwen2_gen
from models.phi3 import phi3_gen
from func.utils import *

def main():
    st.set_page_config(page_title="Анализ текста ИИ", page_icon="🤖")

    st.title("🤖 Анализ текста на следы ИИ и переформулирование")

    st.write("""
    Добро пожаловать в наше приложение! Здесь вы можете проанализировать текст на наличие следов ИИ и переформулировать его для улучшения SEO.
    """)

    tab1, tab2 = st.tabs(["Анализ следов ИИ", "Переформулирование текста"])

    with tab1:
        st.info("""
Модель AI-детектора DistilBERT

Эта модель используется для определения, был ли текст написан человеком или искусственным интеллектом. Она основана на технологии DistilBERT и показывает высокую точность в своих предсказаниях.

Основные результаты модели:
- Точность на обучающем наборе данных: 99.83%
- Точность на тестовом наборе данных: 99.58%

Эта модель помогает анализировать тексты и выявлять, где использовались автоматические генераторы текста.
""")

        st.header("Анализ текста на наличие следов ИИ")
        user_text = st.text_area("Введите текст для анализа")
        if st.button("Анализировать"):
            if user_text:
                results = classify_text(user_text)
                human_score = None
                ai_score = None

                for result in results:
                    label = result['label']
                    score_percent = result['score']
                    
                    if label == 'human':
                        human_score = score_percent
                    elif label == 'AI':
                        ai_score = score_percent
                
                if human_score is not None:
                    st.markdown(
                        f"""
                        <div style="padding: 10px; border-radius: 5px; background-color: green; color: white; font-weight: bold; text-align: center;">
                            Оценка присутствия следов человека (human): {human_score}%
                        </div>
                        <div style="padding: 10px; border-radius: 5px; background-color: red; color: white; font-weight: bold; text-align: center;margin-top: 15px;">
                            Оценка присутствия следов ИИ (AI): {percentage_to_100(human_score)}%
                        </div>
       
       
                        """, unsafe_allow_html=True
                    )
                if ai_score is not None:
                    st.markdown(
                        f"""
                        <div style="padding: 10px; border-radius: 5px; background-color: red; color: white; font-weight: bold; text-align: center;">
                            Оценка присутствия следов ИИ (AI): {ai_score}%
                        </div>
                        
                        <div style="padding: 10px; border-radius: 5px; background-color: green; color: white; font-weight: bold; text-align: center; margin-top: 15px;">
                            Оценка присутствия следов человека (human): {percentage_to_100(ai_score)}%
                        </div>
                        """, unsafe_allow_html=True
                        )
           
            else:
                st.info("Пожалуйста, введите текст для анализа.")


    with tab2:
        st.header("Переформулирование текста")
        user_text = st.text_area("Введите текст для переформулирования")
        rewrite_method = st.selectbox("Выберите метод переформулирования", ["llama3", "qwen2", "Gemma", "phi3", "Цепи Маркова"])
        
        if rewrite_method == "Цепи Маркова":
            st.info("Генерация текста методом цепей Маркова не является лучшим решением для перефразирования. Используйте разумно.")
        
        if st.button("Переформулировать"):
            if user_text:
                
                if rewrite_method == "qwen2":
                    qwen2_text = qwen2_gen(user_text)
                    st.write("Переформулированный текст:")
                
                    st.write(qwen2_text)
                    
                if rewrite_method == "phi3":
                    phi3_text = phi3_gen(user_text)
                    st.write("Переформулированный текст:")
                
                    st.write(phi3_text)
                    
                    
                if rewrite_method == "Gemma":
                    gemma_text = gemma_gen(user_text)
                    st.write("Переформулированный текст:")
                    
                    st.write(gemma_text)
                    
                if rewrite_method == "llama3":
                    llama_text = llama_gen(user_text)
                    st.write("Переформулированный текст:")
                
                    st.write(llama_text)
                    
                    
                if rewrite_method == "Цепи Маркова":
                    rephrased_text = makarov_chain_rewrite(user_text)
                    st.write("Переформулированный текст:")
                    st.write(rephrased_text)
            else:
                st.info("Пожалуйста, введите текст для переформулирования.")

if __name__ == "__main__":
    main()
