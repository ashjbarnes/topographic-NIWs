
View Letter
Date:	Mar 12, 2024
To:	"Ashley James Barnes" ashley.barnes@anu.edu.au
cc:	klymak.jpo@ametsoc.org
From:	"Jody Klymak" Klymak.JPO@ametsoc.org
Subject:	Decision for JPO-D-23-0223
Mar 12, 2024
Ref.: JPO-D-23-0223
Editor Decision

Dear Mr Barnes,

I am now in receipt of all reviews of the manuscript "Topographically-generated near-internal waves as a response to winds over the ocean surface" by Ashley James Barnes; Callum Shakespeare; Andy Hogg; Navid Constantinou.

On the basis of these reviews and my own evaluation, I am sorry to inform you that this manuscript must be rejected from further consideration for publication in the Journal of Physical Oceanography at this time. The reviewers' comments are included below.

Both reviewers felt this was a great study and a very interesting problem. "Rejection" here is solely meant in terms of time needed to adequately address the reviewers' concerns. In particular, it seems you need more simulations with more vertical levels - five levels seems very parsimonious for a problem that is looking at near-inertial wave generation on the sea floor. You will have to justify that choice, and other choices in the modelling (for instance the very large horizontal domain seems like a place that you could borrow resolution from). My intuition is doing these experiments at higher vertical resolution will quantitatively change your results, and lead to substantive revisions.

I strongly encourage you to resubmit, and if and when you do, please include a detailed response to the reviewers. If you persists with using only five vertical levels, please have a strong argument for why that is adequate.

Thank you for the opportunity to consider your work.


Sincerely,

Dr. Jody Klymak
Editor
Journal of Physical Oceanography

....................................................................

REVIEWER COMMENTS

Reviewer #1: The authors explore a new mechanism for topographically generated near-inertial waves forced by surface wind storms and associated pressure adjustments in a simple numerical model. I found the study interesting to read. The authors explore a novel topic that is well worth publishing in JPO once some issues have been addressed. I have two main criticisms that I detail below before I list a number of smaller comments and questions.

1) I am wondering why you do not show hydrostatic pressure if that's the driving force for the topographically generated waves? It would be very nice to see a time series of wind forcing together with hydrostatic pressure (or pressure difference across topography at depth) and induced bottom flow. Maybe also surface elevation - I couldn't quite figure out whether the pressure is completely barotropic or also has a baroclinic component.

2) In a few places the paper reads a bit as if you are trying to oversell this effect, for example, you mention "up to 50% of the corresponding conversion to surface-generated NI waves" which is not wrong but an outlier in your results. I don't think this is necessary! You have a really interesting hypothesis that appears to explain at least some of the upward propagating shear we see all over the place in the ocean. I don't think there is a need to try to make the process bigger than it is.

---

**Detailed Comments**

1: Do you mean near-inertial waves?

15: Consider mentioning frequency of the internal waves.

18: I suggest also highlighting the range of conversion rates you find in your sensitivity studies.

23: Wouldn't it be more straight forward to talk about mixing of waters of different densities than mixing heat and salt between them?

24: Turbulent mixing certainly affects global ocean circulation but wind and surface heat fluxes are at least as important in this regard, please be specific here.

25: It may be good to mention that to our current knowledge turbulent mixing is driven by breaking internal waves.

20: quantity -> quantify

26: delete "that" so the sentence makes sense.

30: found -> find

31: "another type" - be a bit more specific

36: What about purely inertial waves that don't have vertical oscillations?

36: a stratified

44: types of internal waves

55: Both references in this sentence should have

66: What does "principally" mean? It seems this sentence works well without it.

67: "tides" should also be plural here.

113: Only five layers seems very little to me (but I don't have much experience with modeling in general and isopycnal coordinate systems in particular). Could you let me & other readers know why this is enough?

**Already intend to fix**
117: I understand that you want to keep this model simple and thus employ linear stratification (isn't that the same as "equally spaced isopycnals"?) but the mixed layer plays an important role in momentum transfers and is usually present in the ocean. Did you try to run your model including a mixed layer and did you get qualitatively similar results?

**Run with light top layer**

122: I am having a bit of trouble parsing the wind forcing. If y is zero in the center of the domain, how do you get your maximum wind forcing there as shown in Fig. 1? Why does the forcing switch direction (from + to -tau_0) in an instant? Doesn't this introduce quite the shock in your model?

123: So T_tau=12 hours?

136: In practice, near-inertial waves often don't have an initial preferred direction but the beta effect will guide them equatorwards.

162: What do you mean with "along with the corresponding wind work"? I don't see how wind work enters the wave energy calculation in (2). Do you mean you calculate the wind work at the location of the 2D slice?for

175: propagates -> propagate

176: "at the time snapshot" -> rephrase

176: Why is the expected frequency for topographically generated NI waves higher than f?

179: Do you have an explanation for the pattern in Fig. 3a that is first quite asymmetric, then transitions to symmetric around 6 inertial periods, and the goes back to an asymmetric state after about 14 inertial periods?

184: wavenumbers -> wavenumber

193: westwards -> westward

193: It would be good to state the magnitude of the mean flow.

208: Why not be precise and say 6%?

210: I think "surface near-inertial waves" should read "surface-generated near-inertial waves" as your energy calculation integrates over all depth layers.

211: Can you point out what exactly you are referring to in Alford (2020)? I have a feeling that this relates to propagating NI waves in the reference, however, your NI energy calculation is for the whole water column and thus would include the mixed layer.

213: Again, I think you should be precise and change 10% to 6%.

218: Delete "It is clear that".

248: Delete "is".

264: Unclear what "doubled" refers to.

294: Mean flow-generated lee waves appear twice in this sentence.

299: You mention 800m instead of 750m earlier in the text.

303: Wouldn't the amount of shear associated with the wave also depend on the vertical wavenumber?

314: "However, ..." -> the sentence appears to be incomplete.

316: I appreciate these two paragraphs with suggestions for further studies!

351: It would be good to also mention here the average percentage relative to surface-generated NI waves.

---

**Figures**

Fig. 1: I hope this schematic can be a bit larger in the paper, axis labels are tiny. What are the colors of the surfaces shown?

Fig. 2: Would "end of storm" be a better description for panel (a) as 12 hours is exactly when you switch off the wind?

Fig. 3: In (b), can you specify which wavenumber you are showing on the x-axis? The green line label should be "Inertial frequency" as you have frequency, not period, on the y-axis. Colorbar label should be "power spectral density..." with appropriate units.

Fig. 4: Please explain conversion rate in caption.


----------------------------

Reviewer #2: The authors introduce a novel mechanism for internal wave generation resulting from wind-driven oscillatory flows interacting with bottom topography. Utilizing idealized numerical simulations and conducting a series of perturbation experiments, they explore the sensitivity of wave generation across various parameters. While the proposed mechanism appears plausible and could potentially contribute significantly to ocean mixing, the explanation provided by the authors lacks clarity, and the supporting evidence and contextualization within the broader spectrum of oceanic mechanisms are insufficient. Detailed comments regarding these concerns are outlined below. I recommend that the authors address these points comprehensively before I can support the publication of this manuscript.

Major comments:

The proposed mechanism consists of two distinct components: (1) wind generating barotropic oscillatory flow and (2) the subsequent interaction of this flow with topography, leading to the emergence of near-inertial internal waves. I think the explanation of the first component requires further clarification. It is suggested that the authors dedicate a separate section to explain the dynamics governing the generation of barotropic flow. This explanation should be accompanied by model analysis illustrating the response of sea surface height and barotropic flow to wind forcing. Key questions regarding the intensity of the resulting barotropic flow, its frequency characteristics, and its comparative magnitude relative to tidal and geostrophic flows in the ocean remain unaddressed.

**Check how barotropic the flow is. Simply put velocities on plots of bottom flow**

In arguing for the significance of these newly proposed topographic near-inertial waves as a potential energy source for mixing, it is imperative to provide evidence of energy conversion or estimates of expected energy dissipation in conventional units (e.g., W, W/m^2, or W/kg). Such quantification would facilitate comparisons with other mechanisms and observed energy dissipation/mixing estimates.

**Divide energy through by time to get watts. Make clear that I've divided by the hill size**

Regarding line 93, the authors should offer a more explicit explanation of the term 'imprinting on the sea surface.' As above, it is recommended to allocate a separate section to elucidate how wind-induced flows manifest at topography, supplemented with plots depicting the evolution of sea surface height and full-depth mean flows.

**Same as previous review comment**

In Section 2, the chosen number of vertical layers requires justification. Specifically, the rationale behind employing 5 layers with an 800m thickness each needs to be addressed. I am strongly convinced that this resolution may be too coarse to adequately resolve the waves discussed in this manuscript. What is the predicted vertical scale of the waves that the model aims to resolve?

Furthermore, Fig. 2 clearly illustrates:

a. The waves propagating into the interior are primarily represented by only 2 layers per wavelength, one layer for the crest and one for the trough. This coarse vertical resolution raises concerns regarding the model's ability to accurately simulate topographic wave generation.

b. The amplitude of flow response in the bottom layer exceeds that in the layers above by an order of magnitude. This potentially indicates that while the ridge may generate waves of significant amplitude, the inadequate vertical resolution impedes their effective propagation away from the topography. If this is the case, this suggests a substantial underestimation of the importance of the proposed wave generation mechanism by the authors.

These issues underscore the necessity for a thorough re-evaluation of the model's vertical resolution to ensure the robustness of the findings and the accuracy of the proposed mechanism.

Section 2: The selection of parameters and various scales warrants further justification and elucidation. Why were these specific values chosen for the domain size (4000km by 4000km), the ridge width (12.5km), and height (500m), as well as the wind spatial scale (300km)? Additional justification for these choices supported by references is necessary to enhance clarity and understanding.

**1-2 sentences to justify defaults**

Equation (1): The implementation of wind forcing as a series of instantaneous jumps (on, reverse, off) requires clarification. What guides this choice? While one could argue it resembles a 12-hour wind storm, utilizing jumps likely induces a significant transient flow response to abrupt changes in wind. Moreover, the rationale behind choosing a pulse forcing and analyzing essentially a transient spin-down problem for the waves remains unclear. It might be more convincing to simulate continuous oscillatory wind forcing leading to an equilibrated wave response, rather than transient pulses, to support the proposed mechanism and avoid model initialization artifacts.

**They've misunderstood the wind timeseries. Storm is a finite event. Oscillating problem produces a steady state and is a different problem

Line 127 "minimise the induced mean flow": Further explanation is needed. Is the mean flow induced by the wind forcing or by the waves and associated wave stresses? Would a sinusoidal wind stress also induce a mean flow at a steady state? If the mean flow is a physical response to the chosen wind, why should it be minimized?

Line 131 "their corresponding topographic scatter": Do surface-generated waves indeed reach the bottom and scatter at topography? Earlier, it was mentioned that it takes weeks for surface near-inertial waves to reach the bottom. This, in fact, suggests an efficient method of separating topographic near-inertial waves from surface ones. Running simulations with a greater ocean depth (e.g., 10-20km) could keep surface and bottom-generated waves separated for longer, enabling a comprehensive analysis of bottom-generated waves over the entire domain and potentially utilizing more realistic multi-scale topography.

Equation 2 and lines 152-156: While discussing wave energetics is helpful, it's essential to focus on energy flux rather than wave energy. Wind work and energy conversion are typically quantified in energy flux units (e.g., W, W/m^2, and W/kg), facilitating comparison with other mechanisms and observed energy dissipation/mixing estimates. Although the authors chose to examine a transient wind pulse problem and quantify it with the total energy input/conversion in [J], other wave generation mechanisms are usually quantified differently.

**As with previous comment put these units on conversion.**

Line 159 "experience different amounts of wind forcing": This statement is unclear. Both types of waves are generated by the same wind. While they may exist in different parts of the domain, it's unclear why they can't be compared.

**I'll need to explain this more clearly**


Figure 2: The velocity amplitudes of waves that radiate (0.05 mm/s) are remarkably small compared to amplitudes of internal waves contributing to mixing in the ocean (1-10 cm/s). This suggests that the waves might not be adequately resolved, as discussed in the comments on vertical resolution. With such minuscule velocity amplitudes, corresponding energy fluxes, wave shear, and wave energy dissipation are likely negligible, limiting their contribution to mixing in the ocean.

**Should be resolved by the question about energy quantification**

Lines 188-190: While the spectrum in Fig. 3b proves useful for identifying generated modes, the assertion of definitive evidence for internal waves appears redundant. Given the idealized framework of the isopycnal model on an f-plane, which inherently limits dynamics to internal waves, emphasizing their internal wave nature may be unnecessary. Instead, exploring the mechanisms behind wave generation and comparing the spectrum with that from wave generation theory could offer deeper insights.

**Hopefully satisfied with more work on wave generation section**

Lines 210-215: Comparing estimates to other mechanisms or energy dissipation/mixing observations would strengthen the manuscript. Simulations should be designed with realistic scales/parameters to contextualize model results within the ocean. Current simulations produce tiny wave amplitudes, obscuring the validity and relevance of the results for the ocean and diminishing the significance of comparing two wave types within the same simulation.

**Quantified energy flux might be enough to satisfy? Number will be small but that's ok**

The section on perturbation experiments lacks theoretical context to interpret results effectively. If there were a theory predicting these sensitivities, experiments could test it. As presented, the purpose of these perturbation runs isn't clear. They seem to be used to argue that topographic waves can be significant under certain conditions, yet the waves are tiny in the reference simulation, and the criteria for "right circumstances" remain ambiguous. This section could be removed all together and the manuscript could focus on (1) a comprehensive explanation of the mechanism, (2) detailed description of the model design with appropriate resolution and forcing, (3) detailed results from the reference model simulation demonstrating waves and their characteristics, and (4) an energetics discussion of relevance to the ocean.

Minor Comments:

The title should say "near-inertial" waves.

Line 7: Waves technically do not carry momentum but rather energy. They redistribute momentum of the mean flow.

Regarding line 10, consider omitting "isopycnal heights" from the sentence. It seems that the authors primarily discuss the rapid communication of surface height anomalies to the bottom through hydrostatic pressure in the manuscript. While isopycnal height anomalies could also be rapidly communicated to the bottom, generating such anomalies from wind would likely take much longer. Moreover, the role of isopycnal anomalies in driving bottom flows does not seem to be a significant aspect of the narrative. If it is indeed pertinent, it should be elucidated and substantiated more effectively.

**This will be clarified by the timeseries of pressure from surface and internal layers**

Line 16: Please provide explicit clarification regarding the circumstances under discussion.

Line 29: I propose that the authors consider framing their findings as a novel form of near-inertial wave generation rather than categorizing them as a new type of waves. It's important to clarify that near-inertial waves are not novel phenomena, even at topography. What distinguishes this study is the revelation that they are generated by the wind-driven oscillatory flow interacting with topography.

Line 31 "as strong as another type of internal waves": Please specify the type of internal waves being compared.

Regarding line 38, consider adding a citation to a review paper on internal waves at the end of the sentence. The cited papers should not be solely credited for the general result that waves contribute to mixing and momentum transfers.

Line 64: Referring to the flow generating waves as non-wave flow is ambiguous. It's recommended to define the mean flow as the spatial mean in this context and then refer to it as "mean flow".

Lines 66-69: The statement regarding the contribution of internal tides and lee waves compared to surface near-inertial waves should be more precise. Providing details, values, and corresponding references would enhance clarity.

Line 86: Somewhere here, it would be beneficial to acknowledge previous studies showing the generation of near-inertial waves at topography (e.g., Liang and Thurnherr, 2012; Brearley et al., 2013; Hu et al., 2020) through the geostrophic mean flow-topography interaction mechanism. Thus, what is truly new in this study is not the identification of a new wave type, but rather the revelation of the generation mechanism facilitated by the deep wind-driven oscillatory mean flow.

Line 101: To accurately characterize the waves, consider referring to them as "wind-driven topographic near-inertial waves" rather than simply "topographic near-inertial waves" given other potential generation mechanisms of near-inertial waves at topography.

Line 129: Specify the nature of the other near-inertial waves explicitly, as they are pertinent to the discussion and frequently referenced in the manuscript.

Lines 134-136: Parentheses may not be necessary.

Lines 138: Does the term "aforementioned background oscillations" refer to surface near-inertial waves? Please be explicit and maintain consistent terminology throughout the text.

Line 140: While looking along the center line filters out surface-generated near-inertial waves, nonlinear interactions between surface and bottom-generated waves might lead to additional waves along the center line. The separation may not be as clear as implied.

Line 147: It is suggested to define mean flow as a spatially-uniform barotropic oscillation early in the manuscript and consistently refer to it as "mean flow".

Line 185: Consider explaining why the model is not run with a y-periodic domain.

Figure 3: Isopycnal displacements of 4cm are tiny for internal waves.

Table 1: Verify the correctness of values. Should the wind work value be 10^6 J? Provide quantities in [W], [W/m^2], or [W/kg] to facilitate comparison with other mechanisms.

Line 294-295: This sentence needs editing.

Lines 302-304: While near-inertial waves in the ocean are characterized by short vertical scales and strong vertical shear, model results show negligible vertical shear based on Fig. 2, (0.05mm/s)/(1000m)=5e-8 1/s.

Lines 307-308: Provide context for the values 2 N/m^2 and 200km. Are these typical for storms?

__________________________________________________
In compliance with data protection regulations, you may request that we remove your personal registration details at any time. (Use the following URL: https://www.editorialmanager.com/amsjpo/login.asp?a=r). Please contact the publication office if you have any questions.

**Add a section to varying the structure of ocean (20 layers and a mixed layer). Look up the drho that's typically used in models**
