# تصحيح حدود مراقب القياس في التجارب الرئيسية — 1.0.3

هذا الإصدار يصحح بقاء حدود تجربة التوقيت (ست جولات وReference فقط) داخل
مشغّل التجارب الرئيسية. يقرأ مراقب القياس الآن حدود المهمة المجمّدة نفسها.
لا يغيّر النموذج أو الترميز أو البيانات أو المكتبات أو إعدادات التدريب.

## الإجراء الحالي: تحضير حملة مصححة فقط

احفظ `EHR_FL_Main_Experiments_Package_v1_0_3.zip` في Downloads. من PowerShell
عادية، دون Administrator:

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\EHR_FL_Main_Experiments_Package_v1_0_3.zip" -DestinationPath "$env:USERPROFILE\EHR_FL_Main_Experiments_v1_0_3"
```

إذا فشل فك الضغط، توقف قبل أمر التشغيل.

```powershell
$WslExe = "$env:WINDIR\System32\wsl.exe"
$Package = "/mnt/c/Users/RNO/EHR_FL_Main_Experiments_v1_0_3/EHR_FL_Main_Experiments_Package_v1_0_3"
$ParentCampaign = "/home/mai/EHR_FL_Main_v1/campaigns/20260909T071824Z_83a874db"
& $WslExe -d Ubuntu-24.04 --exec /bin/bash "$Package/scripts/run_on_wsl.sh" "$Package" PrepareRepair "$ParentCampaign"
```

المصدر هذه المرة هو **الحملة المنتهية بـ071824Z_83a874db** التي أكملت A ثم
توقفت B. لا تستخدم الحملة الأصلية 040230Z أو محاولة تحضير غير مكتملة.
احتفظ بجميع المجلدات والنتائج السابقة دون حذف أو تعديل. شغّل نسخة واحدة فقط،
مع 4 GiB على الأقل داخل Linux. لا تنزيل أو تثبيت مكتبات في هذا الأمر.

## ماذا يفعل؟

يفحص الحملة السابقة وتقاريرها وخطأ ServerApp المحلي، وينشئ حملة جديدة ينسخ
إليها مصفوفات التطوير والبدايات وقوائم السحب المجمدة مطابقة بالبايت. لا يعيد
التدريب أو المعالجة أو اختيار الجلسات، ولا ينسخ أوزان A أو B المدرّبة.

يفحص رسائل كل جولة مخططة، بما فيها الجولة 7 والجولة 30، عند مراقب الخادم
ومراقب العميل في B وC وحالتي التوزيع. الفحص يستعمل كائنات Flower الأصلية
ورسائل اختبار عامة ونقلاً محلياً اختبارياً: **ليس تدريباً ولا تشغيل SecAgg+
فعلياً ولا قياس تكلفة جديداً**. تبقى فحوص التغليف ونسخ القوالب مطلوبة أيضاً.

رسالة الفحص التجميعي الجديدة:

```text
ALL_20_MEASUREMENT_HORIZON_AND_CONDITION_PROBES_PASS
```

ورسالة اكتمال التحضير:

```text
MAIN_PREPARATION_PASS_NOT_MAIN_TRAINING_OR_TEST
```

## التقرير

بعد العودة إلى سطر PowerShell:

```powershell
explorer.exe "$env:USERPROFILE\EHR_FL_Main_Experiments_v1_0_3\EHR_FL_Main_Experiments_Package_v1_0_3"
```

ارفع أحدث `Main_Preparation_Report_....zip` في المحادثة، سواء نجح أم توقف.
لا تستخدم إذن التدريب القديم مع الحملة الجديدة، ولا تستخدم RetryBlock أو
تغير عدد الجولات والبذور والمكتبات. بعد مراجعة التحضير نربط إطلاق التدريب
بالحملة الجديدة ثم نعيد الكتلة الثلاثية وفق سياسة الإصلاح؛ لا نختار أفضل A
من المحاولات السابقة.

للتفاصيل الإنجليزية راجع `MEASUREMENT_SCOPE_FIX.md`. ملفات الإصلاحات السابقة
في الحزمة محفوظة للتاريخ؛ هذا الملف يحدد أمر التشغيل الحالي.
