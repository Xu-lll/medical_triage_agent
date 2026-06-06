import unittest

from app.agent import MedicalTriageAgent
from app.database import DEFAULT_DB_PATH, init_db


class MedicalTriageAgentTest(unittest.TestCase):
    def setUp(self):
        init_db(DEFAULT_DB_PATH)
        self.agent = MedicalTriageAgent(DEFAULT_DB_PATH, use_llm=False, use_vector=False)

    def test_chest_pain_goes_to_emergency(self):
        result = self.agent.ask("胸痛半小时，出汗，左肩也疼，还有点呼吸困难")
        self.assertIn("急诊科", result.answer)
        self.assertIn("急症", result.answer)

    def test_uti_retrieval(self):
        result = self.agent.ask("最近尿频尿急尿痛，有一点血尿")
        self.assertIn("泌尿外科", result.answer)
        self.assertIn("尿路感染", result.answer)

    def test_common_cold_does_not_go_to_emergency(self):
        result = self.agent.ask("鼻塞流涕打喷嚏，轻微咳嗽，低热一天")
        self.assertIn("可先门诊/线上问诊评估", result.answer)
        self.assertIn("呼吸内科", result.answer)
        self.assertNotIn("建议立即前往急诊科", result.answer)

    def test_single_chest_discomfort_is_not_automatic_emergency(self):
        result = self.agent.ask("饭后胸口灼热反酸嗳气，平卧后更明显")
        self.assertIn("可先门诊/线上问诊评估", result.answer)
        self.assertIn("消化内科", result.answer)
        self.assertNotIn("建议立即前往急诊科", result.answer)

    def test_response_time_is_recorded(self):
        result = self.agent.ask("最近尿频尿急尿痛")
        self.assertGreaterEqual(result.response_time_ms, 0)

    def test_unmatched_query_does_not_fall_back_to_emergency(self):
        result = self.agent.ask("今天感觉有点说不出来的不舒服")
        self.assertIn("全科", result.answer)
        self.assertNotIn("建议立即前往急诊科", result.answer)

    def test_followup_context_is_used(self):
        first = self.agent.ask("鼻塞流涕打喷嚏，低热")
        second = self.agent.ask("3天前才出现症状")
        self.assertIn("收到", second.answer)
        self.assertIn("继续判断", second.answer)
        self.assertNotIn("信息不足，建议补充描述后再评估", second.answer)

    def test_medication_question_gets_safe_conversation(self):
        self.agent.ask("发热咳嗽两天，肌肉酸痛")
        result = self.agent.ask("我该吃什么药")
        self.assertIn("用药", result.answer)
        self.assertIn("不能", result.answer)
        self.assertIn("补充", result.answer)

    def test_general_chat_gets_normal_reply(self):
        result = self.agent.ask("你好")
        self.assertIn("你好", result.answer)
        self.assertIn("哪里不舒服", result.answer)

    def test_bloody_stool_followup_keeps_abdominal_context(self):
        self.agent.ask("腹泻腹痛")
        result = self.agent.ask("存在便血的症状，疼了好几天了")
        self.assertIn("消化内科", result.answer)
        self.assertNotIn("胸痛急诊规则", result.answer)

    def test_llm_mode_without_llm_does_not_use_docs(self):
        result = self.agent.ask("腹泻腹痛", knowledge_mode="llm")
        self.assertEqual([], result.docs)
        self.assertIn("纯 LLM 模式", result.answer)

    def test_department_question_gets_direct_department(self):
        self.agent.ask("我头疼发热怎么办")
        result = self.agent.ask("该去医院哪个诊室")
        self.assertIn("挂", result.answer)
        self.assertTrue("全科" in result.answer or "发热门诊" in result.answer or "呼吸内科" in result.answer)

    def test_no_other_symptoms_does_not_repeat_other_symptom_question(self):
        self.agent.ask("我头疼发热怎么办")
        self.agent.ask("从昨天开始，加重没有好转，发热最影响我")
        result = self.agent.ask("没有其他症状了")
        self.assertNotIn("有没有", result.answer)
        self.assertNotIn("其他症状", result.answer.split("请")[-1])

    def test_oral_blisters_route_to_stomatology(self):
        result = self.agent.ask("我口腔起了很多水泡，有点疼，该去什么诊室")
        self.assertIn("口腔科", result.answer)
        self.assertNotIn("发热门诊", result.answer)


if __name__ == "__main__":
    unittest.main()
