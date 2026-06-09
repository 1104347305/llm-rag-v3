---
schema_version: "2.1"
type: data
entity_type: extraction_audit
knowledge_domain: general
domain: general
title: "平安附加e生保（尊享版）长期医疗保险（费率可调）.pdf 抽取质量审计"
source_files: ["平安附加e生保（尊享版）长期医疗保险（费率可调）.pdf"]
sources: ["平安附加e生保（尊享版）长期医疗保险（费率可调）.pdf"]
confidence: 0.51
status: candidate
needs_review: false
attributes: {"score":51,"pageCount":7,"expectedItemCount":53,"expectedCoverage":0.1320754716981132,"missingCandidateCount":5,"criticalCoverage":0.625,"highCoverage":0.24,"recommendedCoverage":0.3333333333333333,"relationCoverage":0.2857142857142857,"evidenceCoverage":1,"knowledgeGapCount":53,"sourceSignals":{"kind":"table_or_catalog","expectedItemCount":53,"detectedItems":[],"tableLikeRowCount":53}}
created: "2026-06-02"
updated: "2026-06-02"
---

# 平安附加e生保（尊享版）长期医疗保险（费率可调）.pdf 抽取质量审计

综合评分：**51/100**

## 处理概况

- 处理模式：direct
- 原文字符数：4916
- 编译上下文字符数：4916
- 分批数量：1
- 解析置信度：high

## 覆盖率

- 生成业务知识页：7
- 原文预计条目：53
- 原文条目覆盖率：13%
- critical 字段完整率：63%
- high_confidence 字段完整率：24%
- recommended 字段完整率：33%
- 关系覆盖率：29%
- 证据覆盖率：100%
- schema 候选缺失数：5
- 显式知识缺口数：53

## 原文信号

- 文档形态：table_or_catalog
- 检测到的服务/规则条目：未识别
- 表格/清单行信号：53

## 页面字段审计

| 页面 | 类型 | critical | high | recommended | schema关系 | 正文关系 | 证据 | 缺口 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [[平安附加e生保（尊享版）长期医疗保险（费率可调）]] | product | 6/8 | 4/9 | 1/3 | 0 | 0 | 16 | 12 |
| [[情景二：赔付率高于行业平均赔付率-10% 费率调整规则]] | coverage_rule | 0/0 | 0/0 | 0/0 | 0 | 0 | 4 | 4 |
| [[情景三：赔付率高于行业平均赔付率-10% 费率调整规则]] | coverage_rule | 0/0 | 0/0 | 0/0 | 0 | 0 | 4 | 4 |
| [[情景一：赔付率高于行业平均赔付率-10% 费率调整规则]] | coverage_rule | 0/0 | 0/0 | 0/0 | 0 | 0 | 4 | 3 |
| [[自第4年起费率调整规则]] | coverage_rule | 0/0 | 0/0 | 0/0 | 0 | 0 | 6 | 4 |
| [[居家健康服务]] | service_benefit | 2/4 | 1/8 | 1/3 | 3 | 0 | 2 | 13 |
| [[居家管家服务]] | service_benefit | 2/4 | 1/8 | 1/3 | 4 | 0 | 2 | 13 |

## 需要关注的页面

- [[平安附加e生保（尊享版）长期医疗保险（费率可调）]]：missing critical=product_code, regulatory_filing_no；missing high=target_age_range, waiting_period_days, payment_period_options, product_documents, underwriting_basics；schema关系=0；正文关系=0
- [[情景二：赔付率高于行业平均赔付率-10% 费率调整规则]]：missing critical=无；missing high=无；schema关系=0；正文关系=0
- [[情景三：赔付率高于行业平均赔付率-10% 费率调整规则]]：missing critical=无；missing high=无；schema关系=0；正文关系=0
- [[情景一：赔付率高于行业平均赔付率-10% 费率调整规则]]：missing critical=无；missing high=无；schema关系=0；正文关系=0
- [[自第4年起费率调整规则]]：missing critical=无；missing high=无；schema关系=0；正文关系=0
- [[居家健康服务]]：missing critical=service_category, core_value；missing high=eligible_customers, application_process, time_limits, service_provider, coverage_scope, service_limits, compliance_notes；schema关系=3；正文关系=0
- [[居家管家服务]]：missing critical=service_category, core_value；missing high=eligible_customers, application_process, time_limits, service_provider, coverage_scope, service_limits, compliance_notes；schema关系=4；正文关系=0

## 建议

- 本次编译不建议直接演示问答，应先补齐低覆盖页面或重新分批抽取。
- critical 字段缺失偏多，优先检查产品、服务、规则的核心字段。
- high_confidence 字段不足，可能影响 Agent 精准检索和结构化过滤。
- 关系覆盖不足，建议补充 Product-Customer-Method 以及服务-规则-合规链接。
- schema 候选缺失仍存在，建议进入审核队列确认是补页、合并到已有页，还是标记为不处理。
