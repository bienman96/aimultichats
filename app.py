"""
Multi-AI Debate Tool v5
=======================
- 비밀번호 보호
- Supabase 데이터 영구 저장
- 모드별 AI 토론
"""

import streamlit as st
import os
from datetime import datetime
from openai import OpenAI
import anthropic
import google.generativeai as genai
from supabase import create_client, Client

# =============================================================================
# 설정
# =============================================================================

def get_secret(key_name: str, default: str = "") -> str:
    """Streamlit secrets에서 값 로드"""
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
    return os.getenv(key_name, default)

# 비밀번호
APP_PASSWORD = get_secret("APP_PASSWORD", "")

# API 키
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
ANTHROPIC_API_KEY = get_secret("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")

# Supabase 설정
SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")

# AI 모델
GPT_MODEL = "gpt-4.1"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
GEMINI_MODEL = "gemini-2.5-pro-preview-06-05"

# 클라이언트 초기화
openai_client = None
anthropic_client = None
supabase: Client = None

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
if ANTHROPIC_API_KEY:
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =============================================================================
# 비밀번호 보호
# =============================================================================

def check_password():
    """비밀번호 확인"""
    if not APP_PASSWORD:
        return True
    
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if st.session_state.authenticated:
        return True
    
    st.title("🔐 로그인")
    password = st.text_input("비밀번호", type="password")
    
    if st.button("로그인", type="primary"):
        if password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("비밀번호가 틀렸습니다")
    
    return False

# =============================================================================
# Supabase 데이터 함수
# =============================================================================

def get_chat_list():
    """채팅방 목록 가져오기"""
    if not supabase:
        return []
    try:
        response = supabase.table("chats").select("*").order("updated_at", desc=True).execute()
        return response.data or []
    except Exception as e:
        st.error(f"채팅 목록 로드 실패: {e}")
        return []

def load_chat(chat_id: str):
    """채팅방 데이터 로드"""
    if not supabase:
        return None
    try:
        # 채팅 정보
        chat_response = supabase.table("chats").select("*").eq("id", chat_id).single().execute()
        chat = chat_response.data
        
        if not chat:
            return None
        
        # 메시지 로드
        msg_response = supabase.table("messages").select("*").eq("chat_id", chat_id).order("created_at").execute()
        chat["messages"] = msg_response.data or []
        
        # 결론 로드
        con_response = supabase.table("conclusions").select("*").eq("chat_id", chat_id).order("created_at").execute()
        chat["conclusions"] = con_response.data or []
        
        return chat
    except Exception as e:
        st.error(f"채팅 로드 실패: {e}")
        return None

def create_new_chat(name: str, mode: str) -> str:
    """새 채팅방 생성"""
    if not supabase:
        return None
    try:
        chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            "id": chat_id,
            "name": name,
            "mode": mode,
            "system_prompt": get_default_system_prompt(mode),
        }
        supabase.table("chats").insert(data).execute()
        return chat_id
    except Exception as e:
        st.error(f"채팅 생성 실패: {e}")
        return None

def update_chat(chat_id: str, updates: dict):
    """채팅방 정보 업데이트"""
    if not supabase:
        return
    try:
        updates["updated_at"] = datetime.now().isoformat()
        supabase.table("chats").update(updates).eq("id", chat_id).execute()
    except Exception as e:
        st.error(f"채팅 업데이트 실패: {e}")

def delete_chat(chat_id: str):
    """채팅방 삭제 (메시지, 결론도 CASCADE 삭제됨)"""
    if not supabase:
        return
    try:
        supabase.table("chats").delete().eq("id", chat_id).execute()
    except Exception as e:
        st.error(f"채팅 삭제 실패: {e}")

def save_message(chat_id: str, role: str, content: str, ai_name: str = None):
    """메시지 저장"""
    if not supabase:
        return
    try:
        data = {
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "ai_name": ai_name
        }
        supabase.table("messages").insert(data).execute()
        # 채팅방 updated_at 갱신
        update_chat(chat_id, {})
    except Exception as e:
        st.error(f"메시지 저장 실패: {e}")

def save_conclusion(chat_id: str, content: str):
    """결론 저장"""
    if not supabase:
        return
    try:
        data = {
            "chat_id": chat_id,
            "content": content
        }
        supabase.table("conclusions").insert(data).execute()
        update_chat(chat_id, {})
    except Exception as e:
        st.error(f"결론 저장 실패: {e}")

def clear_chat_messages(chat_id: str):
    """채팅방 메시지 초기화"""
    if not supabase:
        return
    try:
        supabase.table("messages").delete().eq("chat_id", chat_id).execute()
        update_chat(chat_id, {})
    except Exception as e:
        st.error(f"초기화 실패: {e}")

def get_default_system_prompt(mode: str) -> str:
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
# API 키 확인
# =============================================================================

def check_api_keys():
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        missing.append("SUPABASE_KEY")
    return missing

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

def build_context_from_messages(messages, max_turns=20):
    recent = messages[-max_turns:] if len(messages) > max_turns else messages
    parts = []
    for msg in recent:
        ai_name = msg.get("ai_name", "")
        content = msg.get("content", "")
        if ai_name:
            parts.append(f"[{ai_name}]: {content}")
        else:
            parts.append(f"[사용자]: {content}")
    return "\n".join(parts)

def build_messages_for_api(messages, max_turns=20):
    recent = messages[-max_turns:] if len(messages) > max_turns else messages
    api_messages = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        ai_name = msg.get("ai_name", "")
        if role == "user":
            api_messages.append({"role": "user", "content": content})
        else:
            prefix = f"[{ai_name}] " if ai_name else ""
            api_messages.append({"role": "assistant", "content": f"{prefix}{content}"})
    return api_messages

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

def run_debate_round(user_message, messages, system_prompt, mode, target_ai=None):
    responses = []
    context = build_context_from_messages(messages)
    api_messages = build_messages_for_api(messages)
    
    current_messages = api_messages + [{"role": "user", "content": user_message}]
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

# 비밀번호 체크
if not check_password():
    st.stop()

# 세션 초기화
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "show_new_chat_form" not in st.session_state:
    st.session_state.show_new_chat_form = False

# API 키 확인
missing_keys = check_api_keys()
if missing_keys:
    st.error(f"⚠️ 설정 누락: {', '.join(missing_keys)}")
    st.info("Streamlit Cloud의 Secrets에 위 항목들을 추가하세요.")
    st.stop()

# 사이드바
with st.sidebar:
    st.title("💬 채팅방")
    
    if st.button("➕ 새 채팅", use_container_width=True, type="primary"):
        st.session_state.show_new_chat_form = True
    
    if st.session_state.show_new_chat_form:
        with st.form("new_chat_form"):
            new_name = st.text_input("이름", placeholder="예: 거북선 게임")
            new_mode = st.selectbox("모드", ["웹소설", "게임개발", "일반토론"])
            mode_ais = {"웹소설": "GPT, Gemini", "게임개발": "Claude, Gemini", "일반토론": "GPT, Claude, Gemini"}
            st.caption(f"AI: {mode_ais[new_mode]}")
            
            c1, c2 = st.columns(2)
            with c1:
                if st.form_submit_button("만들기"):
                    if new_name.strip():
                        new_id = create_new_chat(new_name.strip(), new_mode)
                        if new_id:
                            st.session_state.current_chat_id = new_id
                            st.session_state.show_new_chat_form = False
                            st.rerun()
            with c2:
                if st.form_submit_button("취소"):
                    st.session_state.show_new_chat_form = False
                    st.rerun()
    
    st.divider()
    
    # 채팅 목록
    chat_list = get_chat_list()
    if chat_list:
        for chat in chat_list:
            icon = {"웹소설": "📖", "게임개발": "🎮", "일반토론": "💭"}.get(chat.get("mode", ""), "💬")
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
    else:
        st.caption("채팅방이 없습니다")
    
    st.divider()
    
    if APP_PASSWORD:
        if st.button("🚪 로그아웃", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

# 메인 영역
if st.session_state.current_chat_id:
    chat_data = load_chat(st.session_state.current_chat_id)
    
    if chat_data:
        mode = chat_data.get("mode", "일반토론")
        available_ais = get_available_ais(mode)
        icon = {"웹소설": "📖", "게임개발": "🎮", "일반토론": "💭"}.get(mode, "💬")
        
        c1, c2 = st.columns([4, 1])
        with c1:
            st.title(f"{icon} {chat_data['name']}")
            st.caption(f"AI: {', '.join(available_ais)}")
        with c2:
            if st.button("🔄 대화 초기화"):
                clear_chat_messages(st.session_state.current_chat_id)
                st.rerun()
        
        with st.expander("⚙️ 시스템 프롬프트"):
            new_sys = st.text_area("", chat_data.get("system_prompt", ""), height=120)
            if st.button("저장"):
                update_chat(st.session_state.current_chat_id, {"system_prompt": new_sys})
                st.success("저장됨!")
        
        # 결론 표시
        conclusions = chat_data.get("conclusions", [])
        if conclusions:
            with st.expander(f"📋 결론 ({len(conclusions)}개)"):
                for i, con in enumerate(conclusions):
                    created = con.get("created_at", "")[:16].replace("T", " ")
                    st.markdown(f"**{i+1}.** {created}")
                    st.info(con.get("content", ""))
        
        st.divider()
        
        # 메시지 표시
        messages = chat_data.get("messages", [])
        for msg in messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                ai = msg.get("ai_name", "AI")
                av = {"GPT": "🟢", "Claude": "🟠", "Gemini": "🔵", "System": "💾"}.get(ai, "🤖")
                with st.chat_message("assistant", avatar=av):
                    st.markdown(f"**[{ai}]**")
                    st.write(msg["content"])
        
        # 입력
        if user_input := st.chat_input("입력... (지정: 클로드:, 지피티:, 제미나이: / 저장: <<확정>>)"):
            chat_id = st.session_state.current_chat_id
            system_prompt = chat_data.get("system_prompt", "")
            
            if check_conclusion_trigger(user_input):
                with st.chat_message("user"):
                    st.write(user_input)
                save_message(chat_id, "user", user_input)
                
                summary_prompt = f"토론 정리:\n{build_context_from_messages(messages)}\n\n지시: {user_input}"
                
                with st.spinner("결론 정리..."):
                    if mode == "게임개발":
                        conclusion = call_claude([{"role": "user", "content": summary_prompt}], "토론 정리 전문가")
                    else:
                        conclusion = call_gpt([{"role": "user", "content": summary_prompt}], "토론 정리 전문가")
                
                save_conclusion(chat_id, conclusion)
                save_message(chat_id, "assistant", f"📋 결론 저장됨\n\n{conclusion}", "System")
                
                with st.chat_message("assistant", avatar="💾"):
                    st.success(conclusion)
            else:
                target, actual = parse_target_ai(user_input)
                with st.chat_message("user"):
                    st.write(user_input)
                
                save_message(chat_id, "user", actual)
                
                spinner = f"{target} 답변 중..." if target else f"토론 중... ({', '.join(available_ais)})"
                with st.spinner(spinner):
                    responses = run_debate_round(actual, messages, system_prompt, mode, target)
                
                av_map = {"GPT": "🟢", "Claude": "🟠", "Gemini": "🔵"}
                for ai, resp in responses:
                    with st.chat_message("assistant", avatar=av_map.get(ai, "🤖")):
                        st.markdown(f"**[{ai}]**")
                        st.write(resp)
                    save_message(chat_id, "assistant", resp, ai)
            
            st.rerun()
    else:
        st.error("채팅을 불러올 수 없습니다")
        st.session_state.current_chat_id = None

else:
    st.title("🤖 Multi-AI Debate Tool")
    st.markdown("""
### 모드별 AI
| 모드 | AI |
|---|---|
| 📖 웹소설 | GPT, Gemini |
| 🎮 게임개발 | Claude, Gemini |
| 💭 일반토론 | 전원 |

### 사용법
- 그냥 입력 → 전원 토론
- `클로드:`, `지피티:`, `제미나이:` → 지정 AI만
- `<<확정>>` → 결론 저장
    """)
    
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📖 웹소설", use_container_width=True):
            new_id = create_new_chat("새 웹소설", "웹소설")
            if new_id:
                st.session_state.current_chat_id = new_id
                st.rerun()
    with c2:
        if st.button("🎮 게임개발", use_container_width=True):
            new_id = create_new_chat("새 게임", "게임개발")
            if new_id:
                st.session_state.current_chat_id = new_id
                st.rerun()
    with c3:
        if st.button("💭 일반토론", use_container_width=True):
            new_id = create_new_chat("새 토론", "일반토론")
            if new_id:
                st.session_state.current_chat_id = new_id
                st.rerun()
