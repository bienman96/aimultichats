"""
Multi-AI Debate Tool v3
=======================
모드별 AI 토론 시스템

AI 구성:
- 웹소설: GPT, Gemini (2명)
- 게임개발: Claude, Gemini (2명)  
- 일반토론: GPT, Claude, Gemini (3명)

기능:
- 여러 채팅방 관리
- AI 지정 호출 (클로드:, 지피티:, 제미나이:)
- <<확정>>, <<결론>>으로 토론 결과 저장
"""

import streamlit as st
import os
import json
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import anthropic
import google.generativeai as genai

# =============================================================================
# 설정
# =============================================================================

def get_api_key(key_name: str) -> str:
    """Streamlit secrets -> 환경변수 순서로 API 키 로드"""
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
    return os.getenv(key_name, "")

OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")
ANTHROPIC_API_KEY = get_api_key("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = get_api_key("GOOGLE_API_KEY")

GPT_MODEL = "gpt-4.1"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
GEMINI_MODEL = "gemini-2.5-pro-preview-06-05"

openai_client = None
anthropic_client = None

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
if ANTHROPIC_API_KEY:
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

DATA_DIR = Path("chat_data")
DATA_DIR.mkdir(exist_ok=True)

def check_api_keys():
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    return missing

# =============================================================================
# 채팅방 관리
# =============================================================================

def get_chat_list():
    chats = []
    if DATA_DIR.exists():
        for f in DATA_DIR.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    chats.append({
                        "id": f.stem,
                        "name": data.get("name", f.stem),
                        "mode": data.get("mode", "일반토론"),
                        "updated": data.get("updated", "")
                    })
            except:
                pass
    chats.sort(key=lambda x: x.get("updated", ""), reverse=True)
    return chats

def load_chat(chat_id):
    filepath = DATA_DIR / f"{chat_id}.json"
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_chat(chat_id, data):
    data["updated"] = datetime.now().isoformat()
    filepath = DATA_DIR / f"{chat_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def create_new_chat(name, mode):
    chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = {
        "id": chat_id, "name": name, "mode": mode,
        "created": datetime.now().isoformat(),
        "updated": datetime.now().isoformat(),
        "system_prompt": get_default_system_prompt(mode),
        "messages": [], "debate_history": [], "conclusions": []
    }
    save_chat(chat_id, data)
    return chat_id

def delete_chat(chat_id):
    filepath = DATA_DIR / f"{chat_id}.json"
    if filepath.exists():
        filepath.unlink()

def get_default_system_prompt(mode):
    if mode == "웹소설":
        return """당신은 한국 웹소설 전문가입니다.
주인공의 단계적 성장과 독자에게 주는 기대감/대리만족을 중시합니다.
플롯, 캐릭터, 성장 설계, 명장면 구성에 대해 전문적인 의견을 제시합니다."""
    elif mode == "게임개발":
        return """당신은 게임 개발 전문가입니다.
알고리즘 설계, 코드 구현, 최적화에 대해 전문적인 의견을 제시합니다.
GameMaker Studio 2 (GML), Godot (GDScript), Python 등에 능숙합니다.
코드는 반드시 ```gml, ```python, ```gdscript 등 코드 블록으로 감싸서 출력하세요."""
    else:
        return """당신은 다양한 주제에 대해 깊이 있는 토론이 가능한 AI입니다."""

# =============================================================================
# AI 호출
# =============================================================================

def call_gpt(messages, system_prompt=""):
    if not openai_client:
        return "[GPT 오류] API 키 없음"
    try:
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        response = openai_client.chat.completions.create(
            model=GPT_MODEL, messages=full_messages, max_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[GPT 오류] {str(e)}"

def call_claude(messages, system_prompt=""):
    if not anthropic_client:
        return "[Claude 오류] API 키 없음"
    try:
        response = anthropic_client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4000,
            system=system_prompt if system_prompt else "당신은 도움이 되는 AI입니다.",
            messages=messages
        )
        return response.content[0].text
    except Exception as e:
        return f"[Claude 오류] {str(e)}"

def call_gemini(prompt, context=""):
    if not GOOGLE_API_KEY:
        return "[Gemini 오류] API 키 없음"
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"[Gemini 오류] {str(e)}"

# =============================================================================
# 토론 로직
# =============================================================================

def build_context_from_history(history, max_turns=20):
    recent = history[-max_turns:] if len(history) > max_turns else history
    parts = []
    for msg in recent:
        ai_name = msg.get("ai_name", "")
        content = msg.get("content", "")
        if ai_name:
            parts.append(f"[{ai_name}]: {content}")
        else:
            parts.append(f"[사용자]: {content}")
    return "\n".join(parts)

def build_messages_for_api(history, max_turns=20):
    recent = history[-max_turns:] if len(history) > max_turns else history
    messages = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        ai_name = msg.get("ai_name", "")
        if role == "user":
            messages.append({"role": "user", "content": content})
        else:
            prefix = f"[{ai_name}] " if ai_name else ""
            messages.append({"role": "assistant", "content": f"{prefix}{content}"})
    return messages

def parse_target_ai(user_input):
    prefixes = {
        "클로드:": "Claude", "claude:": "Claude", "Claude:": "Claude",
        "지피티:": "GPT", "gpt:": "GPT", "GPT:": "GPT", "챗지피티:": "GPT",
        "제미나이:": "Gemini", "gemini:": "Gemini", "Gemini:": "Gemini", "제미니:": "Gemini",
    }
    for prefix, ai_name in prefixes.items():
        if user_input.strip().startswith(prefix):
            return ai_name, user_input.strip()[len(prefix):].strip()
    return None, user_input

def check_conclusion_trigger(text):
    triggers = ["<<확정>>", "<<결론>>", "<<저장>>", "<<정리>>"]
    return any(t in text for t in triggers)

def get_available_ais(mode):
    if mode == "웹소설":
        return ["GPT", "Gemini"]
    elif mode == "게임개발":
        return ["Claude", "Gemini"]
    else:
        return ["GPT", "Claude", "Gemini"]

def run_debate_round(user_message, history, system_prompt, mode, target_ai=None):
    responses = []
    context = build_context_from_history(history)
    messages = build_messages_for_api(history)
    
    current_messages = messages + [{"role": "user", "content": user_message}]
    current_context = context + f"\n[사용자]: {user_message}"
    
    available_ais = get_available_ais(mode)
    ai_list_str = ", ".join(available_ais)
    
    debate_system = system_prompt + f"""

당신은 {len(available_ais)}명의 AI({ai_list_str})가 함께 토론하는 세션에 참여중입니다.
다른 AI들의 의견을 참고하되, 당신만의 관점을 명확히 제시하세요.
간결하고 핵심적인 답변을 하세요.
"""
    
    if target_ai:
        if target_ai == "GPT":
            resp = call_gpt(current_messages, debate_system)
            responses.append(("GPT", resp))
        elif target_ai == "Claude":
            resp = call_claude(current_messages, debate_system)
            responses.append(("Claude", resp))
        elif target_ai == "Gemini":
            resp = call_gemini(user_message, current_context + "\n\n" + debate_system)
            responses.append(("Gemini", resp))
    else:
        accumulated_context = current_context
        accumulated_messages = current_messages
        
        for i, ai_name in enumerate(available_ais):
            if i == 0:
                if ai_name == "GPT":
                    resp = call_gpt(accumulated_messages, debate_system)
                elif ai_name == "Claude":
                    resp = call_claude(accumulated_messages, debate_system)
                elif ai_name == "Gemini":
                    resp = call_gemini(user_message, accumulated_context + "\n\n" + debate_system)
            else:
                prev_responses = "\n".join([f"- {n}: {r[:800]}" for n, r in responses])
                enhanced_system = debate_system + f"\n\n지금까지의 토론:\n{prev_responses}\n\n이에 대해 당신의 의견을 제시하세요."
                
                if ai_name == "GPT":
                    resp = call_gpt(accumulated_messages, enhanced_system)
                elif ai_name == "Claude":
                    resp = call_claude(accumulated_messages, enhanced_system)
                elif ai_name == "Gemini":
                    gemini_prompt = f"이전 토론:\n{accumulated_context}\n\n사용자 질문: {user_message}"
                    resp = call_gemini(gemini_prompt, debate_system)
            
            responses.append((ai_name, resp))
            accumulated_context += f"\n[{ai_name}]: {resp}"
            accumulated_messages.append({"role": "assistant", "content": f"[{ai_name}] {resp}"})
    
    return responses

# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(page_title="Multi-AI Debate", page_icon="🤖", layout="wide")

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "show_new_chat_form" not in st.session_state:
    st.session_state.show_new_chat_form = False

# API 키 확인
missing_keys = check_api_keys()
if missing_keys:
    st.error(f"⚠️ API 키 누락: {', '.join(missing_keys)}")
    with st.expander("🔑 API 키 설정 방법", expanded=True):
        st.markdown("""
`.streamlit/secrets.toml` 파일 생성:
```toml
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
GOOGLE_API_KEY = "AIza..."
```
        """)
    st.stop()

# 사이드바
with st.sidebar:
    st.title("💬 채팅방")
    
    if st.button("➕ 새 채팅 시작", use_container_width=True, type="primary"):
        st.session_state.show_new_chat_form = True
    
    if st.session_state.show_new_chat_form:
        with st.form("new_chat_form"):
            st.subheader("새 채팅방")
            new_name = st.text_input("이름", placeholder="예: 거북선 게임")
            new_mode = st.selectbox("모드", ["웹소설", "게임개발", "일반토론"])
            mode_ais = {"웹소설": "GPT, Gemini", "게임개발": "Claude, Gemini", "일반토론": "GPT, Claude, Gemini"}
            st.caption(f"참여 AI: {mode_ais[new_mode]}")
            
            c1, c2 = st.columns(2)
            with c1:
                if st.form_submit_button("만들기", use_container_width=True):
                    if new_name.strip():
                        new_id = create_new_chat(new_name.strip(), new_mode)
                        st.session_state.current_chat_id = new_id
                        st.session_state.show_new_chat_form = False
                        st.rerun()
            with c2:
                if st.form_submit_button("취소", use_container_width=True):
                    st.session_state.show_new_chat_form = False
                    st.rerun()
    
    st.divider()
    
    for chat in get_chat_list():
        icon = {"웹소설": "📖", "게임개발": "🎮", "일반토론": "💭"}.get(chat["mode"], "💬")
        c1, c2 = st.columns([5, 1])
        with c1:
            is_active = st.session_state.current_chat_id == chat["id"]
            if st.button(f"{icon} {chat['name']}", key=f"c_{chat['id']}", 
                        use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.current_chat_id = chat["id"]
                st.rerun()
        with c2:
            if st.button("🗑️", key=f"d_{chat['id']}"):
                delete_chat(chat["id"])
                if st.session_state.current_chat_id == chat["id"]:
                    st.session_state.current_chat_id = None
                st.rerun()
    
    st.divider()
    with st.expander("🤖 모드별 AI"):
        st.markdown("| 모드 | AI |\n|---|---|\n| 📖웹소설 | GPT, Gemini |\n| 🎮게임 | Claude, Gemini |\n| 💭토론 | 전원 |")

# 메인
if st.session_state.current_chat_id:
    chat_data = load_chat(st.session_state.current_chat_id)
    
    if chat_data:
        mode = chat_data["mode"]
        available_ais = get_available_ais(mode)
        icon = {"웹소설": "📖", "게임개발": "🎮", "일반토론": "💭"}.get(mode, "💬")
        
        c1, c2 = st.columns([4, 1])
        with c1:
            st.title(f"{icon} {chat_data['name']}")
            st.caption(f"AI: {', '.join(available_ais)}")
        with c2:
            if st.button("🔄 초기화"):
                chat_data["messages"] = []
                chat_data["debate_history"] = []
                save_chat(st.session_state.current_chat_id, chat_data)
                st.rerun()
        
        with st.expander("⚙️ 시스템 프롬프트"):
            new_sys = st.text_area("", chat_data.get("system_prompt", ""), height=120)
            if st.button("저장"):
                chat_data["system_prompt"] = new_sys
                save_chat(st.session_state.current_chat_id, chat_data)
                st.success("저장됨!")
        
        with st.expander("📁 참조 파일"):
            uploaded = st.file_uploader("파일", type=["txt", "md", "py", "gml", "json", "gd"])
            ref_content = ""
            if uploaded:
                ref_content = uploaded.read().decode("utf-8")
                st.success(f"로드: {uploaded.name}")
        
        if chat_data.get("conclusions"):
            with st.expander(f"📋 결론 ({len(chat_data['conclusions'])}개)"):
                for i, con in enumerate(chat_data["conclusions"]):
                    st.markdown(f"**{i+1}.** {con.get('timestamp', '')}")
                    st.info(con.get("content", ""))
        
        st.divider()
        
        for msg in chat_data.get("messages", []):
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                ai = msg.get("ai_name", "AI")
                av = {"GPT": "🟢", "Claude": "🟠", "Gemini": "🔵", "System": "💾"}.get(ai, "🤖")
                with st.chat_message("assistant", avatar=av):
                    st.markdown(f"**[{ai}]**")
                    st.write(msg["content"])
        
        if user_input := st.chat_input("입력... (지정: 클로드:, 지피티:, 제미나이: / 저장: <<확정>>)"):
            full_sys = chat_data.get("system_prompt", "")
            if ref_content:
                full_sys += f"\n\n[참조]\n{ref_content}"
            
            if check_conclusion_trigger(user_input):
                with st.chat_message("user"):
                    st.write(user_input)
                chat_data["messages"].append({"role": "user", "content": user_input})
                
                summary_prompt = f"토론 정리:\n{build_context_from_history(chat_data.get('debate_history', []))}\n\n지시: {user_input}"
                
                with st.spinner("결론 정리..."):
                    if mode == "게임개발":
                        conclusion = call_claude([{"role": "user", "content": summary_prompt}], "토론 정리 전문가")
                    else:
                        conclusion = call_gpt([{"role": "user", "content": summary_prompt}], "토론 정리 전문가")
                
                chat_data.setdefault("conclusions", []).append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "content": conclusion
                })
                chat_data["messages"].append({"role": "assistant", "ai_name": "System", "content": f"📋 결론 저장됨\n\n{conclusion}"})
                save_chat(st.session_state.current_chat_id, chat_data)
                
                with st.chat_message("assistant", avatar="💾"):
                    st.success(conclusion)
            else:
                target, actual = parse_target_ai(user_input)
                with st.chat_message("user"):
                    st.write(user_input)
                
                chat_data["messages"].append({"role": "user", "content": user_input})
                chat_data["debate_history"].append({"role": "user", "content": actual})
                
                spinner = f"{target} 답변 중..." if target else f"토론 중... ({', '.join(available_ais)})"
                with st.spinner(spinner):
                    responses = run_debate_round(actual, chat_data.get("debate_history", []), full_sys, mode, target)
                
                av_map = {"GPT": "🟢", "Claude": "🟠", "Gemini": "🔵"}
                for ai, resp in responses:
                    with st.chat_message("assistant", avatar=av_map.get(ai, "🤖")):
                        st.markdown(f"**[{ai}]**")
                        st.write(resp)
                    chat_data["messages"].append({"role": "assistant", "ai_name": ai, "content": resp})
                    chat_data["debate_history"].append({"role": "assistant", "ai_name": ai, "content": resp})
                
                save_chat(st.session_state.current_chat_id, chat_data)
            st.rerun()

else:
    st.title("🤖 Multi-AI Debate Tool v3")
    st.markdown("""
### 모드별 AI 구성
| 모드 | AI | 용도 |
|---|---|---|
| 📖 웹소설 | GPT, Gemini | 플롯, 성장 설계 |
| 🎮 게임개발 | Claude, Gemini | 알고리즘, 코딩 |
| 💭 일반토론 | GPT, Claude, Gemini | 범용 |

### 사용법
- **그냥 입력**: 모드별 AI 전원 토론
- **`클로드:`, `지피티:`, `제미나이:`**: 해당 AI만
- **`<<확정>>`**: 결론 저장
    """)
    
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📖 웹소설\n(GPT, Gemini)", use_container_width=True):
            st.session_state.current_chat_id = create_new_chat("새 웹소설", "웹소설")
            st.rerun()
    with c2:
        if st.button("🎮 게임개발\n(Claude, Gemini)", use_container_width=True):
            st.session_state.current_chat_id = create_new_chat("새 게임", "게임개발")
            st.rerun()
    with c3:
        if st.button("💭 일반토론\n(전원)", use_container_width=True):
            st.session_state.current_chat_id = create_new_chat("새 토론", "일반토론")
            st.rerun()
