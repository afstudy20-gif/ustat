import { X, BookOpen, FlaskConical, Brain, Settings, ShieldCheck, HelpCircle, Code, CheckCircle2, Info } from "lucide-react";
import { useState } from "react";

type TabId = "quickstart" | "hypothesis" | "advanced" | "specialized" | "rhub";

export default function HelpModal({ onClose }: { onClose: () => void }) {
  const [activeTab, setActiveTab] = useState<TabId>("quickstart");

  const tabs = [
    { id: "quickstart", label: "Quick Start / Veri", icon: BookOpen },
    { id: "hypothesis", label: "Hypothesis Tests", icon: FlaskConical },
    { id: "advanced",   label: "Causal & Regression", icon: Brain },
    { id: "specialized",label: "EFA, Bayes & Meta", icon: Settings },
    { id: "rhub",       label: "R Replication Hub", icon: Code },
  ] as const;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl flex flex-col h-[650px] max-h-[90vh] overflow-hidden animate-in fade-in zoom-in duration-200">
        
        {/* Header */}
        <div className="bg-slate-900 text-white px-6 py-4 flex items-center justify-between flex-shrink-0">
          <div className="flex items-center gap-2">
            <div className="bg-indigo-500 p-1.5 rounded-lg">
              <HelpCircle size={18} className="text-white" />
            </div>
            <div>
              <h2 className="font-bold text-sm tracking-tight">uSTAT Yardım & Analiz Kılavuzu</h2>
              <p className="text-[10px] text-slate-400 font-mono">uSTAT Help Center & Interactive Tutorial</p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors p-1 hover:bg-slate-800 rounded-lg cursor-pointer"
          >
            <X size={18} />
          </button>
        </div>

        {/* Inner Content Area */}
        <div className="flex-1 flex min-h-0">
          
          {/* Sidebar Navigation */}
          <div className="w-56 bg-slate-50 border-r border-slate-200 flex flex-col p-3 gap-1 flex-shrink-0 overflow-y-auto">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider px-2.5 pb-2">
              Kategoriler / Sections
            </p>
            {tabs.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-xs font-semibold transition-all text-left cursor-pointer ${
                  activeTab === id
                    ? "bg-indigo-600 text-white shadow-md shadow-indigo-100"
                    : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"
                }`}
              >
                <Icon size={14} className={activeTab === id ? "text-white" : "text-slate-400"} />
                {label}
              </button>
            ))}
            
            <div className="mt-auto p-2 bg-indigo-50 rounded-xl border border-indigo-100">
              <p className="text-[10px] font-bold text-indigo-900 uppercase">💡 İpucu / Tip</p>
              <p className="text-[9px] text-indigo-700 mt-0.5 leading-relaxed">
                Her paneldeki <span className="font-semibold">ⓘ</span> veya soru işareti simgelerine yaklaşarak detaylı klinik ipuçlarını okuyabilirsiniz.
              </p>
            </div>
          </div>

          {/* Tab Panels */}
          <div className="flex-1 p-6 overflow-y-auto bg-white font-sans text-xs text-slate-700 space-y-4">
            
            {activeTab === "quickstart" && (
              <div className="space-y-4">
                <div className="border-b pb-2">
                  <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                    🚀 Hızlı Başlangıç & Veri Hazırlığı <span className="text-xs font-normal text-slate-400 font-mono">| Quick Start</span>
                  </h3>
                </div>

                <div className="space-y-3">
                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">1. Dosya Yükleme (Data Upload)</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Excel (<span className="font-mono">.xlsx</span>), CSV, SPSS (<span className="font-mono">.sav</span>) veya TSV dosyalarınızı sürükleyip bırakarak yükleyebilirsiniz. SPSS etiketleriniz ve değişken açıklamalarınız otomatik olarak okunur.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">2. Değişken Türleri & Ölçekler (Variable Kinds)</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        uSTAT değişkenleri otomatik olarak sınıflandırır. Bir değişkeni <span className="text-indigo-600 font-semibold">Numeric</span> (Sürekli) veya <span className="text-teal-600 font-semibold">Categorical</span> (Kategorik) olarak değiştirmek için Data sekmesindeki değişken başlığının yanındaki etiketlere (badge) tıklamanız veya **Data Dictionary** panelini açmanız yeterlidir.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">3. Veri Filtreleme (Active Filters)</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Data sekmesindeki **Filter** butonuna tıklayarak alt gruplar (subsets) oluşturabilirsiniz. Aktif filtre uygulandığında tüm analizler otomatik olarak bu alt gruba göre filtrelenir (başlıkta turuncu badge görünür).
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "hypothesis" && (
              <div className="space-y-4">
                <div className="border-b pb-2">
                  <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                    🧪 Hipotez & Kategorik Testler <span className="text-xs font-normal text-slate-400 font-mono">| Hypothesis Tests</span>
                  </h3>
                </div>

                <div className="space-y-3">
                  <div className="flex gap-2 bg-indigo-50/50 p-3 rounded-xl border border-indigo-100/50">
                    <Info size={16} className="text-indigo-600 flex-shrink-0 mt-0.5" />
                    <p className="text-indigo-800 leading-relaxed text-[11px]">
                      <strong>Otomatik Dağılım Kararı (Auto Mode):</strong> uSTAT hipotez testlerinde otomatik dağılım kontrolü yapar. n ≤ 2000 ise **Shapiro-Wilk**, n &gt; 2000 ise çarpıklık (skewness) ve **Lilliefors Kolmogorov-Smirnov** testleri ile normallik sınanır. Dağılım normal ise parametrik (t-test, ANOVA), değilse otomatik olarak non-parametrik testler (Mann-Whitney U, Kruskal-Wallis) seçilir.
                    </p>
                  </div>

                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">T-Testi & Varyans Analizi (ANOVA)</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Bağımsız iki grup karşılaştırması için Independent t-test (veya Mann-Whitney U), ikiden fazla grup için One-way ANOVA (veya Kruskal-Wallis) uygulanır. Tekrarlı ölçümler için Paired t-test veya Repeated Measures ANOVA sekmesini kullanabilirsiniz.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">Kategorik İlişki Testleri (Categorical)</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        İki kategorik değişken arasındaki oranları karşılaştırmak için Ki-Kare (Chi-square) veya küçük örneklemlerde Fisher's Exact testini uygulayın. Cochran-Armitage testini ise doz-cevap veya sıralı kategoriler arasındaki trendleri incelemek için kullanabilirsiniz.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">Hiyerarşik Uç Nokta Testi (Gatekeeping)</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Çoklu hipotezleri (örneğin birincil ve ikincil sonlanım noktalarını) aileler halinde sıralayarak ve Bonferroni/Hochberg/Holm ağırlıklarını dağıtarak hata payını (FWER) koruyabilirsiniz.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "advanced" && (
              <div className="space-y-4">
                <div className="border-b pb-2">
                  <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                    🧠 Regresyon & Nedensel Çıkarım <span className="text-xs font-normal text-slate-400 font-mono">| Causal & Regression</span>
                  </h3>
                </div>

                <div className="space-y-3">
                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">Regresyon Modelleri & Çoklu Bağlantı (Regression & VIF)</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Lineer (sürekli çıktılar), Lojistik (ikili çıktılar), Firth Penalized Lojistik (nadir olaylar ve ayrışma sorunları için), Poisson ve Cox PH (sağkalım) modellerini eğitebilirsiniz. Tüm modellerde otomatik olarak çoklu bağlantıyı test eden **VIF** (Variance Inflation Factor) değerleri hesaplanır.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">Eğilim Skoru Eşleştirmesi (Propensity Score Matching - PSM)</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Tedavi ve kontrol grupları arasındaki karıştırıcı faktörleri (confounders) eşitleyerek bire bir veya 1:N eşleşmiş kohortlar kurar. Love Plot ve SMD tablosu ile dengelenmeyi doğrular. Sonuçtaki <span className="font-semibold text-indigo-600">"View & Analyze Matched Cohort"</span> butonu ile tüm uygulamayı eşleşmiş hasta listesine kilitleyebilir ve backtracking (Geri Dön) butonuyla orijinal sete geri dönebilirsiniz.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">Ters Olasılık Ağırlıklandırması (IPTW Weighting)</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Eşleştirme yerine, her hastayı propensity skoruna göre ağırlıklandırarak (ATE veya ATT estimandları) tüm veri kümesini dengeler. Ağırlıklı kohortu uygulamaya kilitleyerek tüm analizlerinizde survey-weight mantığında ağırlıklı tahminler yürütebilirsiniz.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "specialized" && (
              <div className="space-y-4">
                <div className="border-b pb-2">
                  <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                    ⚙️ İleri & Özel Analiz Yöntemleri <span className="text-xs font-normal text-slate-400 font-mono">| EFA, Bayes & Meta</span>
                  </h3>
                </div>

                <div className="space-y-3">
                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">Açıklayıcı Faktör Analizi & PCA (Factor Analysis)</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Ölçek ve anket verilerinizin yapısını ortaya çıkarmak için KMO ve Bartlett küresellik testleri ile veri uygunluğunu ölçün. Varimax (dik) veya Promax (eğik) döndürmelerle faktör yüklerini hesaplayıp interaktif Scree Plot ve Biplot grafikleri oluşturabilirsiniz.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">Bayesyen Hipotez Sınama (Bayesian Statistics)</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Klasik p-değerlerine bağlı kalmaksızın, alternatif (H₁) ve boş (H₀) hipotezlerin kanıt gücünü **JZS Bayes Faktörü** (BF₁₀ / BF₀₁) ile test edin. Önsel (Prior) Cauchy eğrisi ile sonsal (Posterior) dağılımın üst üste bindiği grafikleri ve Savage-Dickey yoğunluk oranlarını inceleyin.
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">Meta-Analiz & Yayın Yanlılığı (Meta-Analysis)</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Sabit ve rastgele etkiler modellerini (DL & Paule-Mandel) kullanarak çalışmaları havuzlayın. Egger ve Begg testleri ile yayın yanlılığını (publication bias) sınayın ve Trim-and-Fill metoduyla eksik olabilecek çalışmaları tamamlayın.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === "rhub" && (
              <div className="space-y-4">
                <div className="border-b pb-2">
                  <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                    💻 R Replikasyon Hub & Raporlama <span className="text-xs font-normal text-slate-400 font-mono">| R Replication Hub</span>
                  </h3>
                </div>

                <div className="space-y-3 bg-indigo-950 text-indigo-100 p-4 rounded-2xl border border-indigo-900 shadow-lg">
                  <div className="flex gap-2.5 items-start">
                    <Info size={18} className="text-indigo-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <h4 className="font-bold text-sm text-white">Birebir R Replikasyon Betiği</h4>
                      <p className="text-xs text-indigo-300 mt-1 leading-relaxed">
                        uSTAT'ta yaptığınız her analiz, veri filtreleme veya modelleme arka planda kronolojik olarak kaydedilir. R Replication Hub simgesine tıklayarak, yaptığınız tüm işlemlerin RStudio üzerinde **birebir aynısını çoğaltacak (replicate edecek)** temiz, optimize edilmiş bir R betiğini (<span className="font-mono text-white bg-indigo-900/60 px-1 py-0.5 rounded">.R dosyası</span>) tek tıkla indirebilirsiniz.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="space-y-3 pt-2">
                  <div className="flex gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-slate-800">Yayın Hazır Bulgular (Methods Appendix)</p>
                      <p className="text-slate-500 leading-relaxed mt-0.5">
                        Tüm analizlerinizi bitirdiğinizde R betiğinin yanı sıra makalenizin Yöntem (Methods) bölümüne doğrudan ekleyebileceğiniz, kullanılan yazılım sürümlerini, tohum (seed) değerlerini ve analiz açıklamalarını içeren akademik bir Word belgesi (DOCX) oluşturabilirsiniz.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

          </div>

        </div>

        {/* Footer */}
        <div className="bg-slate-50 border-t border-slate-200 px-6 py-3 flex items-center justify-between flex-shrink-0 text-[10px] text-slate-500 font-medium">
          <div className="flex items-center gap-1.5">
            <ShieldCheck size={13} className="text-emerald-600" />
            <span>Tüm verileriniz yerel tarayıcı oturumunda işlenir ve sunucuya sadece hesaplama için anonim olarak gönderilir.</span>
          </div>
          <span className="font-semibold text-slate-400">uSTAT v2.6.0 Kılavuz</span>
        </div>

      </div>
    </div>
  );
}
