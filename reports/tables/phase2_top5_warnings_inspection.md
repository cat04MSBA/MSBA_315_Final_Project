# Phase 2 — Top 5 films by `parse_warning_count`: raw XML vs parsed output

This report dumps source XML alongside the parser's output for the five films with the highest warning counts in the corpus. Its purpose is to confirm that the parser's recovery rules produced sensible output in the worst empirical cases.

Each entry shows: the film and its top-line counts, the first five warnings emitted during parsing, and for each of the first two scenes named in those warnings, the raw XML alongside the parsed `Scene` object's output.


## 1. 'Iron Man_2008' (`tt0371746`) — 114 warnings

- `n_scenes`: 148
- `n_dialogue_lines`: 1,013
- `n_unique_characters`: 97
- `total_dialogue_chars`: 53,163
- `mean_dialogue_line_length`: 52.5


### First 5 warnings

- scene 3: character 'RHODEY' followed by another <character>
- scene 3: character '© 2007 MARVEL STUDIOS, INC.' had no following <dialogue>
- scene 5: character '© 2007 MARVEL STUDIOS, INC.' had no following <dialogue>
- scene 5: character '© 2007 MARVEL STUDIOS, INC.' had no following <dialogue>
- scene 7: character '© 2007 MARVEL STUDIOS, INC.' had no following <dialogue>

### Scene 3 — raw XML

```xml
<scene>
    <stage_direction>INT. HUMMER - CONTINUOUS</stage_direction>
    <scene_description>Three Airmen, kids with battle-worn faces. Crammed in there with them is a Man in an expensive suit, who looks tele- ported from Beverly Hills. He is, of course, genius inventor and billionaire, TONY STARK. In his hand is a drink tumbler of vodka.</scene_description>
    <character>TONY</character>
    <dialogue>Oh, I get it. You guys aren't allowed to talk. Is that it? Are you not allowed to talk?</dialogue>
    <scene_description>One Airman grins, fidgeting with his orange NY Mets watch.</scene_description>
    <character>JIMMY</character>
    <dialogue>No. We're allowed to talk.</dialogue>
    <character>TONY</character>
    <dialogue>Oh. I see. So it's personal.</dialogue>
    <character>RAMIREZ</character>
    <dialogue>I think they're intimidated.</dialogue>
    <character>TONY</character>
    <dialogue>Good God, you're a woman.</dialogue>
    <scene_description>The others try to compress laughs.</scene_description>
    <character>TONY</character>
    <dialogue>I, honestly, I couldn't have called that.</dialogue>
    <parenthetical>(after silence)</parenthetical>
    <dialogue>I would apologize, but isn't that what we're going for here? I saw you as a soldier first.</dialogue>
    <character>JIMMY</character>
    <dialogue>I have a question, sir.</dialogue>
    <character>TONY</character>
    <dialogue>Please.</dialogue>
    <scene_description>NO DUPLICATION WITHOUT MARVEL'S WRITTEN CONSENT. CONTINUED:</scene_description>
    <character>JIMMY</character>
    <dialogue>Is it true you're twelve for twelve with last years Maxim cover girls?</dialogue>
    <character>TONY</character>
    <dialogue>Excellent question. Yes and no. March and I had a schedule conflict but, thankfully, the Christmas cover was twins. Anyone else? You, with the hand up.</dialogue>
    <character>PRATT</character>
    <dialogue>It's a little embarrassing.</dialogue>
    <character>TONY</characte
... (truncated)
```

### Scene 3 — parsed output

```python
stage_direction:   'INT. HUMMER - CONTINUOUS'
scene_description: 'Three Airmen, kids with battle-worn faces. Crammed in there with them is a Man in an expensive suit, who looks tele- ported from Beverly Hills. He is, of course, genius inventor and billionaire, TONY '
n_dialogue_units:  19
First 8 dialogue_units:
  ('TONY', "Oh, I get it. You guys aren't allowed to talk. Is that it? Are you not allowed t")
  ('JIMMY', "No. We're allowed to talk.")
  ('TONY', "Oh. I see. So it's personal.")
  ('RAMIREZ', "I think they're intimidated.")
  ('TONY', "Good God, you're a woman.")
  ('TONY', "I, honestly, I couldn't have called that.")
  ('TONY', "I would apologize, but isn't that what we're going for here? I saw you as a sold")
  ('JIMMY', 'I have a question, sir.')
```

### Scene 5 — raw XML

```xml
<scene>
    <stage_direction>INT. INSURGENT CAVE - AFGHANISTAN - DAY</stage_direction>
    <scene_description>Tony snaps awake. He's tied to a chair, bloody rags covering his chest. Two Insurgents flank a DV camera. Behind Tony -- A line of armed hooded men and a banner showing ten interlocked rings. The Leader, a huge Choori knife in one hand, reads rhetoric (in Dari) for the camera. PUSH IN ON - THE DV CAMERA VIEWFINDER: until the image of a desperate Tony breaks up into pixel chaos. CUT TO: CREDITS OVER A FULL SCREEN FILM REEL: The attack on Pearl Harbor. FDR gives an impassioned speech.</scene_description>
    <character>NARRATOR (RHODEY'S VOICE)</character>
    <dialogue>December 7, 1941: the day the world changed forever.</dialogue>
    <scene_description>NO DUPLICATION WITHOUT MARVEL'S WRITTEN CONSENT. CONTINUED:</scene_description>
    <character>NARRATOR (RHODEY'S VOICE)</character>
    <dialogue>President Roosevelt declares the United States will build fifty thousand planes to fight the armies of Hirohito and Hitler--</dialogue>
    <scene_description>S.S. Officers goose-step through Paris.</scene_description>
    <character>NARRATOR</character>
    <dialogue>Although no such capacity to build existed...</dialogue>
    <scene_description>1940s L.A., an unassuming hangar reads: "STARK INDUSTRIES."</scene_description>
    <character>NARRATOR</character>
    <dialogue>Howard Stark, founder of the fledgling Stark Industries, answers his call to duty --</dialogue>
    <scene_description>A Young Howard Stark shakes FDR's hand.</scene_description>
    <character>NARRATOR</character>
    <dialogue>And builds not fifty, but a hundred thousand planes.</dialogue>
    <scene_description>An airfield covered in B-29s. Stark bombers in flight, strewing bombs and paratroopers across the sky.</scene_description>
    <character>NARRATOR</character>
    <dialogue>Later, Stark's work on the Manhattan project makes the end of the war possible.</dialogue>
    <scene_description
... (truncated)
```

### Scene 5 — parsed output

```python
stage_direction:   'INT. INSURGENT CAVE - AFGHANISTAN - DAY'
scene_description: "Tony snaps awake. He's tied to a chair, bloody rags covering his chest. Two Insurgents flank a DV camera. Behind Tony -- A line of armed hooded men and a banner showing ten interlocked rings. The Lead"
n_dialogue_units:  17
First 8 dialogue_units:
  ("NARRATOR (RHODEY'S VOICE)", 'December 7, 1941: the day the world changed forever.')
  ("NARRATOR (RHODEY'S VOICE)", 'President Roosevelt declares the United States will build fifty thousand planes ')
  ('NARRATOR', 'Although no such capacity to build existed...')
  ('NARRATOR', 'Howard Stark, founder of the fledgling Stark Industries, answers his call to dut')
  ('NARRATOR', 'And builds not fifty, but a hundred thousand planes.')
  ('NARRATOR', "Later, Stark's work on the Manhattan project makes the end of the war possible.")
  ('NARRATOR', 'Stark Industries would go on to contribute to every major weapons system through')
  ('NARRATOR', "But Howard Stark's greatest achievement would come in 1973--")
```

## 2. 'Cat People_1942' (`tt0034587`) — 62 warnings

- `n_scenes`: 128
- `n_dialogue_lines`: 540
- `n_unique_characters`: 57
- `total_dialogue_chars`: 33,351
- `mean_dialogue_line_length`: 61.8


### First 5 warnings

- scene 1: character 'THE CAT PEOPLE' had no following <dialogue>
- scene 1: character 'EVEN AS FOG CONTINUES TO LIE IN THE VALLEYS, SO DOES ANCIENT' followed by another <character>
- scene 1: character 'SIN CLING TO THE LOW PLACES, THE DEPRESSIONS IN THE WORLD' followed by another <character>
- scene 1: character 'CONSCIOUSNESS.' followed by another <character>
- scene 1: scene ended with character 'SIGMUND FREUD' and no following <dialogue>

### Scene 1 — raw XML

```xml
<scene>
    <scene_description>brought to you by The Val Lewton Screenplay Collection</scene_description>
    <character>THE CAT PEOPLE</character>
    <scene_description>Original Screen Play  by DeWitt Bodeen  The RKO trademark FADES OFF, leaving a black screen, in the center of which are two slits of pale light. These move closer until we see that they are a pair of cat's eyes. Over these mysteriously blinking lights the title is SUPERIMPOSED.</scene_description>
    <character>DISSOLVE</character>
    <dialogue>A misty OUT-OF-FOCUS SHOT of a black panther pacing behind cage bars. Over this come the production credits. A pale fog rises over the shot of the black panther, and over it Is SUPERIMPOSED the following quotation:</dialogue>
    <character>EVEN AS FOG CONTINUES TO LIE IN THE VALLEYS, SO DOES ANCIENT</character>
    <character>SIN CLING TO THE LOW PLACES, THE DEPRESSIONS IN THE WORLD</character>
    <character>CONSCIOUSNESS.</character>
    <character>SIGMUND FREUD</character>
  </scene>
  
```

### Scene 1 — parsed output

```python
stage_direction:   ''
scene_description: 'brought to you by The Val Lewton Screenplay Collection Original Screen Play  by DeWitt Bodeen  The RKO trademark FADES OFF, leaving a black screen, in the center of which are two slits of pale light. '
n_dialogue_units:  6
First 8 dialogue_units:
  ('THE CAT PEOPLE', '')
  ('DISSOLVE', 'A misty OUT-OF-FOCUS SHOT of a black panther pacing behind cage bars. Over this ')
  ('EVEN AS FOG CONTINUES TO LIE IN THE VALLEYS, SO DOES ANCIENT', '')
  ('SIN CLING TO THE LOW PLACES, THE DEPRESSIONS IN THE WORLD', '')
  ('CONSCIOUSNESS.', '')
  ('SIGMUND FREUD', '')
```

### Scene 2 — raw XML

```xml
<scene>
    <stage_direction>EXT. ZOO PROMENADE - PARK - AFTERNOON</stage_direction>
    <scene_description>As the last word of the quotation FADES from the screen, the fog clears, the caged leopard comes into full focus, and we see that it is an actual leopard behind actual bars. Over the scene is the wheezy music of the Triumphal March from "Aida," as played on a hand, organ. This is playing in the distance, and we do not see the organ-grinder until later. The CAMERA DRAWS BACK to show a young artist sitting before the cage on a campstool with a drawing portfolio in her hand. She is presumably sketching the panther, although her drawing is not shown, and we do not see the features of the girl's face. The girl picks up the drawing and holds it off, weighing its values. It evidently does not meet with her approval, for she wads the drawing Into a ball and turns to look for a place to throw the waste paper. We see her face. It is heart-shaped, demure, even a little naive. She is small, young,and very beautiful. In one hand the wad of waste paper is poised, ready to throw into a container. INSERT WASTEPAPER BASKET as Irena sees it. It is a rather fancy container in the shape of a tree trunk.</scene_description>
    <character>OLIVER</character>
    <dialogue>Yes. (then continuing in the same tone)</dialogue>
    <dialogue>"Sometimes whoever seeks abroad may find Thee sitting careless on a granary floor, Thy hair soft lifted... (ponders, as if trying to remember) Thy hair soft lifted..."</dialogue>
    <character>ALICE</character>
    <parenthetical>(snapping her compact shut)</parenthetical>
    <dialogue>Reminds me. I have a date with the hairdresser.</dialogue>
    <character>OLIVER</character>
    <dialogue>What a way to spend a Saturday afternoon!</dialogue>
    <character>ALICE</character>
    <dialogue>The business girl's holiday</dialogue>
    <character>OLIVER</character>
    <dialogue>You should've minded your mother and eaten more bread crusts. You'd have cu
... (truncated)
```

### Scene 2 — parsed output

```python
stage_direction:   'EXT. ZOO PROMENADE - PARK - AFTERNOON'
scene_description: 'As the last word of the quotation FADES from the screen, the fog clears, the caged leopard comes into full focus, and we see that it is an actual leopard behind actual bars. Over the scene is the whee'
n_dialogue_units:  14
First 8 dialogue_units:
  ('OLIVER', 'Yes. (then continuing in the same tone)')
  ('OLIVER', '"Sometimes whoever seeks abroad may find Thee sitting careless on a granary floo')
  ('ALICE', 'Reminds me. I have a date with the hairdresser.')
  ('OLIVER', 'What a way to spend a Saturday afternoon!')
  ('ALICE', "The business girl's holiday")
  ('OLIVER', "You should've minded your mother and eaten more bread crusts. You'd have curly h")
  ('ALICE', 'Thank you for lunch. See you at the office Monday.')
  ('IRENA', 'Thank you.')
```

## 3. 'Batman: Mask of the Phantasm_1993' (`tt0106364`) — 56 warnings

- `n_scenes`: 76
- `n_dialogue_lines`: 856
- `n_unique_characters`: 264
- `total_dialogue_chars`: 54,634
- `mean_dialogue_line_length`: 63.8


### First 5 warnings

- scene 1: character 'CHUCKIE' followed by another <character>
- scene 1: character 'CHUCKIE' followed by another <character>
- scene 1: character 'CHUCKIE' followed by another <character>
- scene 2: character 'CHUCKIE' had no following <dialogue>
- scene 4: scene ended with character 'AUDIENCE (0. S.)' and no following <dialogue>

### Scene 1 — raw XML

```xml
<scene>
    <character>BATMAN</character>
    <dialogue>HOKE VIDEO</dialogue>
    <scene_description>"Masks" Story by: Alan Burnett Screenplay by: Alan Burnett Paul Dini Martin Pasko Michael Reaves Final draft:</scene_description>
    <character>12/21/92</character>
    <dialogue>r =di:)</dialogue>
    <scene_description>WARNER BROS. ANIMATION, INC.</scene_description>
    <character>BATMAN</character>
    <dialogue>HOME VIDEO</dialogue>
    <scene_description>"Masks"</scene_description>
    <character>FADE IN:</character>
    <dialogue>EXT. GOTHAM CITY - THE S~.DY LADY CASINO - NIGET</dialogue>
    <dialogue>Above the entrance a seventy-foot, neon Rita Hayworth lookalike seductively posed with one hand on her .hip, the other behind her head, winks at the traffic below. You can almost hear, "Put the blame on Mame, boys ... "</dialogue>
    <character>0~ "RITA'S" HEAD</character>
    <dialogue>BATMAN circles behind her head on a bat glider and lands silently</dialogue>
    <character>~</character>
    <dialogue>~ i -· on the roof.</dialogue>
    <scene_description>'.'\ ·.,:~~-,•</scene_description>
    <character>ON ROOFTOP</character>
    <dialogue>He steps from the glider and looks around, making sure no one is up there with him. He moves to a certain spot and pulls out a device which he aims at the roof. It fires a dart into the roof with a THWUMP. The portion sticking out of the roof looks like a small antenna.</dialogue>
    <character>CLOSE ON BATMAN</character>
    <dialogue>Batman pulls an earphone from his belt which he puts to his ear. We hear FILTERED VOICES from below ...</dialogue>
    <character>CHUCKIE (V. 0.)</character>
    <parenthetical>(filtered)</parenthetical>
    <dialogue>Take a good look, boys ...</dialogue>
    <character>INT. THE SUITE BELOW - CONTINUOUS</character>
    <dialogue>It's a plush garish HOTEL SUITE. Marble fountain, gold lame wallpaper ... the room they give Sinatra. A fifty-ish, squat racketeer, CHUCKIE SOL (Joe Pesci in a goo
... (truncated)
```

### Scene 1 — parsed output

```python
stage_direction:   ''
scene_description: '"Masks" Story by: Alan Burnett Screenplay by: Alan Burnett Paul Dini Martin Pasko Michael Reaves Final draft: WARNER BROS. ANIMATION, INC. "Masks" \'.\'\\ ·.,:~~-,• -.. ..:.•· Chuckie plucks it from the '
n_dialogue_units:  41
First 8 dialogue_units:
  ('BATMAN', 'HOKE VIDEO')
  ('12/21/92', 'r =di:)')
  ('BATMAN', 'HOME VIDEO')
  ('FADE IN:', 'EXT. GOTHAM CITY - THE S~.DY LADY CASINO - NIGET')
  ('FADE IN:', 'Above the entrance a seventy-foot, neon Rita Hayworth lookalike seductively pose')
  ('0~ "RITA\'S" HEAD', 'BATMAN circles behind her head on a bat glider and lands silently')
  ('~', '~ i -· on the roof.')
  ('ON ROOFTOP', 'He steps from the glider and looks around, making sure no one is up there with h')
```

### Scene 2 — raw XML

```xml
<scene>
    <stage_direction>EXT.. TOP FLOOR OF PARKING STRUCTURE</stage_direction>
    <scene_description>Open air. Very few cars. Chuckie reaches the top of the stairs. He gasps for air for a second, then moves on.</scene_description>
    <character>CHUCKIE</character>
    <parenthetical>(a couple of gasps)</parenthetical>
    <scene_description>DOWNSHOT ON STAIRWELL - ON PHANTASM Following, looking up as he climbs. Face heading TOWARD CAMERA. Should feel like a ghost rising on mist. ON CAR Chuckie opens the driver's door and heaves the valise into the back seat. ON BACK SEAT The case lands next to a thermos-size canister planted on the floor. We hear the car door SLAM. ON CHUCKIE sitting in the front seat, sweating bullets. He turns the ignition. The motor CHURNS, not turning over.</scene_description>
    <character>ON THE STAIRWELL</character>
    <dialogue>:r Phantasm rises from the stairwell on a cloud of mist and heads for</dialogue>
    <scene_description>the car.</scene_description>
    <character>INSIDE CAR - BACK ON CHUCKIE</character>
    <dialogue>The car finally STARTS, to his relief. He looks at the approaching Phantasm as he puts his car in gear.</dialogue>
    <character>CHUCKIE</character>
    <dialogue>This time I got you, you son of a</dialogue>
    <character>EXT. CAR</character>
    <dialogue>With a SQUEAL of tires, the car takes off.</dialogue>
    <character>ON PHANTASM</character>
    <dialogue>who stops in the center of the parking lot as the lights from Chuckie's car bear down. With a wave of his arms Phantasm completely envelops himself in mist once again.</dialogue>
    <character>WIDER</character>
    <dialogue>a split second later Chuckie plows right through the mist, missing Phantasm entirely.</dialogue>
    <character>ON CHUCKIE</character>
    <dialogue>looking back through the rear view mirror with a how-in-the-hell- did-I-miss-him expression.</dialogue>
    <character>BACK ON THE MIST</character>
    <dialogue>There's still a sect
... (truncated)
```

### Scene 2 — parsed output

```python
stage_direction:   'EXT.. TOP FLOOR OF PARKING STRUCTURE'
scene_description: 'Open air. Very few cars. Chuckie reaches the top of the stairs. He gasps for air for a second, then moves on. DOWNSHOT ON STAIRWELL - ON PHANTASM Following, looking up as he climbs. Face heading TOWAR'
n_dialogue_units:  12
First 8 dialogue_units:
  ('CHUCKIE', '')
  ('ON THE STAIRWELL', ':r Phantasm rises from the stairwell on a cloud of mist and heads for')
  ('INSIDE CAR - BACK ON CHUCKIE', 'The car finally STARTS, to his relief. He looks at the approaching Phantasm as h')
  ('CHUCKIE', 'This time I got you, you son of a')
  ('EXT. CAR', 'With a SQUEAL of tires, the car takes off.')
  ('ON PHANTASM', "who stops in the center of the parking lot as the lights from Chuckie's car bear")
  ('WIDER', 'a split second later Chuckie plows right through the mist, missing Phantasm enti')
  ('ON CHUCKIE', 'looking back through the rear view mirror with a how-in-the-hell- did-I-miss-him')
```

## 4. 'Fletch_1985' (`tt0089155`) — 53 warnings

- `n_scenes`: 95
- `n_dialogue_lines`: 1,112
- `n_unique_characters`: 134
- `total_dialogue_chars`: 53,447
- `mean_dialogue_line_length`: 48.1


### First 5 warnings

- scene 3: character 'CREASY AND FLETCH' followed by another <character>
- scene 3: character 'MASTER' followed by another <character>
- scene 11: character 'STANWYK' followed by another <character>
- scene 11: character 'STANWYK' followed by another <character>
- scene 11: character 'STANWYK' followed by another <character>

### Scene 3 — raw XML

```xml
<scene>
    <stage_direction>INT. "FAT SAM�S" - DAY</stage_direction>
    <scene_description>Seated just inside the stand on a folding aluminum chair is a chubby man in his late thirties. He’s wearing a stained valour sweat suit and a cap. This is Fat Sam. He’s a dealer. Seated on the sand next to him is Fletch, a rangy man, early thirties, in jeans and a Magic Johnson T-shirt, nodding idly on a battered Casio music machine which he treats lovingly. This is the source of the title music.</scene_description>
    <character>FLETCH</character>
    <dialogue>So what do you figure?</dialogue>
    <character>FAT SAM</character>
    <dialogue>No idea.</dialogue>
    <character>FLETCH</character>
    <dialogue>No idea at all?</dialogue>
    <character>FAT SAM</character>
    <dialogue>Okay. Some idea.</dialogue>
    <character>FLETCH</character>
    <dialogue>Like when?</dialogue>
    <character>FAT SAM</character>
    <dialogue>Like tonight.</dialogue>
    <character>FLETCH</character>
    <dialogue>For sure?</dialogue>
    <character>FAT SAM</character>
    <dialogue>No, not for sure. When it comes, it comes. You gonna want some shit?</dialogue>
    <character>FLETCH</character>
    <dialogue>I think I’d rather have drugs.</dialogue>
    <character>FAT SAM</character>
    <parenthetical>(shakes head and smiles)</parenthetical>
    <dialogue>Fletch...</dialogue>
    <character>FLETCH</character>
    <dialogue>Sorry. I find a little humor really brightens things up around here, don’t you?</dialogue>
    <scene_description>A young junkie with a black eye – Gummy – passes.</scene_description>
    <character>GUMMY</character>
    <dialogue>Hi Sam. Hi Fletch.</dialogue>
    <character>FLETCH</character>
    <dialogue>Hi Gummy. How’s the eye?</dialogue>
    <character>GUMMY</character>
    <dialogue>It’s okay. The cops did it.</dialogue>
    <character>FLETCH</character>
    <dialogue>I know.</dialogue>
    <character>GUMMY</character>
    <dialogue>They busted me last week.</d
... (truncated)
```

### Scene 3 — parsed output

```python
stage_direction:   'INT. "FAT SAM\x80�S" - DAY'
scene_description: 'Seated just inside the stand on a folding aluminum chair is a chubby man in his late thirties. He’s wearing a stained valour sweat suit and a cap. This is Fat Sam. He’s a dealer. Seated on the sand ne'
n_dialogue_units:  54
First 8 dialogue_units:
  ('FLETCH', 'So what do you figure?')
  ('FAT SAM', 'No idea.')
  ('FLETCH', 'No idea at all?')
  ('FAT SAM', 'Okay. Some idea.')
  ('FLETCH', 'Like when?')
  ('FAT SAM', 'Like tonight.')
  ('FLETCH', 'For sure?')
  ('FAT SAM', 'No, not for sure. When it comes, it comes. You gonna want some shit?')
```

### Scene 11 — raw XML

```xml
<scene>
    <stage_direction>INT. LIBRARY - DAY</stage_direction>
    <scene_description>Massive fireplace. Everything built in teak. Fletch enters, and Stanwyk closes the door behind them.</scene_description>
    <character>FLETCH</character>
    <dialogue>Ahh, the library. Masculine but sensitive.</dialogue>
    <scene_description>Stanwyk wordlessly goes behind the desk</scene_description>
    <character>FLETCH</character>
    <dialogue>Really, I love what you've done with the place. Must have cost you... hundreds.</dialogue>
    <scene_description>Stanwyk turns, looks out a pair of French doors behind his desk, then turns back.</scene_description>
    <character>STANWYK</character>
    <dialogue>Here's my proposition, Mr. Nugent.</dialogue>
    <character>FLETCH</character>
    <dialogue>I'm all ears.</dialogue>
    <character>STANWYK</character>
    <dialogue>I want you to murder me.</dialogue>
    <character>FLETCH</character>
    <dialogue>Even garrulous Fletch is stopped in his tracks by this remark, uttered in the most business-like manner.</dialogue>
    <character>STANWYK</character>
    <character>STANWYK</character>
    <dialogue>Here. On Thursday. I'd like you to shoot me dead.</dialogue>
    <character>FLETCH</character>
    <dialogue>He just stares, barely breathing.</dialogue>
    <character>STANWYK</character>
    <character>STANWYK</character>
    <dialogue>The reason I ask you to do me this service is that I am facing a long, painful, and most certain death. You see, I have bone cancer. I don't know if you know anything about bone cancer.</dialogue>
    <character>FLETCH</character>
    <dialogue>He shakes his head.</dialogue>
    <character>STANWYK</character>
    <character>STANWYK</character>
    <dialogue>It doesn't get any worse than that. Just eats you up, bit by bit.</dialogue>
    <character>FLETCH</character>
    <dialogue>Finally regains the gift of speech.</dialogue>
    <character>FLETCH</character>
    <dialogue>You don't look sick, M
... (truncated)
```

### Scene 11 — parsed output

```python
stage_direction:   'INT. LIBRARY - DAY'
scene_description: 'Massive fireplace. Everything built in teak. Fletch enters, and Stanwyk closes the door behind them. Stanwyk wordlessly goes behind the desk Stanwyk turns, looks out a pair of French doors behind his '
n_dialogue_units:  74
First 8 dialogue_units:
  ('FLETCH', 'Ahh, the library. Masculine but sensitive.')
  ('FLETCH', "Really, I love what you've done with the place. Must have cost you... hundreds.")
  ('STANWYK', "Here's my proposition, Mr. Nugent.")
  ('FLETCH', "I'm all ears.")
  ('STANWYK', 'I want you to murder me.')
  ('FLETCH', 'Even garrulous Fletch is stopped in his tracks by this remark, uttered in the mo')
  ('STANWYK', '')
  ('STANWYK', "Here. On Thursday. I'd like you to shoot me dead.")
```

## 5. 'Monsters University_2013' (`tt1453405`) — 53 warnings

- `n_scenes`: 132
- `n_dialogue_lines`: 1,241
- `n_unique_characters`: 127
- `total_dialogue_chars`: 45,509
- `mean_dialogue_line_length`: 36.7


### First 5 warnings

- scene 2: character 'FIRE STUDENT' had no following <dialogue>
- scene 3: character 'YOUNG MIKE' followed by another <character>
- scene 19: character 'BIG STUDENT #1' followed by another <character>
- scene 19: character 'STUDENTS' followed by another <character>
- scene 20: character 'SULLEY' had no following <dialogue>

### Scene 2 — raw XML

```xml
<scene>
    <stage_direction>EXT. NEIGHBORHOOD - DAY</stage_direction>
    <scene_description>A bird lands on the ground. It pecks at something, then the head stays up and another head pecks at the ground. It turns and we see that it has two heads. It squawks and flies off.... A SCHOOL BUS makes its way down the street. Singing can be heard from the kids inside. Pan up to reveal "Frighton Elementary" on the side of the bus.</scene_description>
    <character>KIDS</character>
    <parenthetical>(singing)</parenthetical>
    <dialogue>The neck bone's connected to the: head bone. The head bone's connected to the: horn bone. The horn bone's right above the...wing bones.</dialogue>
    <parenthetical>(laughing)</parenthetical>
    <dialogue>The bus pulls into a parking lot.</dialogue>
    <dialogue>The bus doors open and a THIRD GRADE CLASS OF MONSTER KIDS pour through, pushing and yelling and laughing and being generally chaotic.</dialogue>
    <scene_description>RAHR!</scene_description>
    <character>KID #2</character>
    <dialogue>Ahh!</dialogue>
    <character>KID #1</character>
    <dialogue>I scared you!</dialogue>
    <character>KID #2</character>
    <parenthetical>(laughing)</parenthetical>
    <dialogue>No you didn't!</dialogue>
    <character>MRS. GRAVES</character>
    <dialogue>Okay, remember our field trip rules everyone: No pushing, no biting, and no fire-breathing.</dialogue>
    <scene_description>One of the kids breathes fire on one of his friends.</scene_description>
    <character>FIRE STUDENT</character>
    <parenthetical>(breathing fire)</parenthetical>
    <scene_description>RAHR! A TEACHER MONSTER, MRS. GRAVES, stands over him, giving him a stern look.</scene_description>
    <character>MRS. GRAVES</character>
    <dialogue>What did I just say?</dialogue>
    <parenthetical>(sigh)</parenthetical>
    <dialogue>18, 19...? Okay, we're missing one. Who are we missing?</dialogue>
    <scene_description>ON THE CLOSED BUS DOORS. A little green hand 
... (truncated)
```

### Scene 2 — parsed output

```python
stage_direction:   'EXT. NEIGHBORHOOD - DAY'
scene_description: 'A bird lands on the ground. It pecks at something, then the head stays up and another head pecks at the ground. It turns and we see that it has two heads. It squawks and flies off.... A SCHOOL BUS mak'
n_dialogue_units:  25
First 8 dialogue_units:
  ('KIDS', "The neck bone's connected to the: head bone. The head bone's connected to the: h")
  ('KIDS', 'The bus pulls into a parking lot.')
  ('KIDS', 'The bus doors open and a THIRD GRADE CLASS OF MONSTER KIDS pour through, pushing')
  ('KID #2', 'Ahh!')
  ('KID #1', 'I scared you!')
  ('KID #2', "No you didn't!")
  ('MRS. GRAVES', 'Okay, remember our field trip rules everyone: No pushing, no biting, and no fire')
  ('FIRE STUDENT', '')
```

### Scene 3 — raw XML

```xml
<scene>
    <stage_direction>INT. MONSTERS, INC. - HALLWAY</stage_direction>
    <scene_description>Mrs. Graves's class is met by a monster TOUR GUIDE.</scene_description>
    <character>MI TOUR GUIDE</character>
    <dialogue>Now stay close together, we're entering a very dangerous area.</dialogue>
    <scene_description>The field trip is entering a scare floor.</scene_description>
    <character>MI TOUR GUIDE</character>
    <dialogue>Welcome to the scare floor.</dialogue>
    <scene_description>The students are in awe as they see the scare floor.</scene_description>
    <character>KIDS</character>
    <parenthetical>(walla)</parenthetical>
    <dialogue>Whoa!</dialogue>
    <character>MI TOUR GUIDE</character>
    <dialogue>This is where we collect the scream energy to power our whole world. And can anyone tell me whose job it is to go get that scream?</dialogue>
    <character>KIDS</character>
    <dialogue>Scarers!</dialogue>
    <character>MI TOUR GUIDE</character>
    <dialogue>That's right! Now which one of you can give me the scariest roar?</dialogue>
    <character>KIDS</character>
    <parenthetical>(raising hands)</parenthetical>
    <dialogue>Ooh! Ooh! Me! Me!</dialogue>
    <scene_description>Mike has his hand raised too.</scene_description>
    <character>YOUNG MIKE</character>
    <dialogue>Ooh, sir! Right here, little green guy at 2 o'clock!</dialogue>
    <character>KID #1</character>
    <dialogue>Roar!</dialogue>
    <character>KID #2</character>
    <dialogue>No, no, it's like this! Raahhr!</dialogue>
    <character>YOUNG MIKE</character>
    <dialogue>Hey guys, watch this one!</dialogue>
    <scene_description>RAAAWRR!</scene_description>
    <character>YOUNG MIKE</character>
    <dialogue>Hey, hey, I got a really goo-</dialogue>
    <scene_description>BRAAAR! GRAAAAHR! RAAAAAOOWRRRRRRRR! The kids turn around.</scene_description>
    <character>KID #1</character>
    <dialogue>Whoa...</dialogue>
    <character>KIDS</character>
    <dialogue>Who
... (truncated)
```

### Scene 3 — parsed output

```python
stage_direction:   'INT. MONSTERS, INC. - HALLWAY'
scene_description: "Mrs. Graves's class is met by a monster TOUR GUIDE. The field trip is entering a scare floor. The students are in awe as they see the scare floor. Mike has his hand raised too. RAAAWRR! BRAAAR! GRAAAA"
n_dialogue_units:  41
First 8 dialogue_units:
  ('MI TOUR GUIDE', "Now stay close together, we're entering a very dangerous area.")
  ('MI TOUR GUIDE', 'Welcome to the scare floor.')
  ('KIDS', 'Whoa!')
  ('MI TOUR GUIDE', 'This is where we collect the scream energy to power our whole world. And can any')
  ('KIDS', 'Scarers!')
  ('MI TOUR GUIDE', "That's right! Now which one of you can give me the scariest roar?")
  ('KIDS', 'Ooh! Ooh! Me! Me!')
  ('YOUNG MIKE', "Ooh, sir! Right here, little green guy at 2 o'clock!")
```