import asyncio
from powerpoint_openwebui_tool import Tools

async def main():
    tools = Tools()
    # Create a new presentation
    result = await tools.create_presentation("Demo PowerPoint Präsentation")
    print(result)
    # Get the generated presentation ID
    prs_id = None
    for key in tools.presentations.keys():
        if key.startswith("demo_powerpoint"):
            prs_id = key
            break
    if not prs_id:
        print("Presentation ID not found.")
        return

    # Add a title slide with subtitle
    result = await tools.add_title_slide(prs_id, "Demo PowerPoint Präsentation", "Dies ist eine Unterüberschrift.")
    print(result)

    # Add a section slide
    result = await tools.add_section_slide(prs_id, "Erster Abschnitt", "Sehr wichtig")
    print(result)

    # Add a content slide
    result = await tools.add_content_slide(prs_id, "Einleitung", "Sehr wichtig", [
        "Dies ist die erste Folie mit Stichpunkten.",
        "Weitere Informationen folgen hier.",
        "Letzter Punkt der Einleitung."
    ])
    print(result)

    # Add another section
    result = await tools.add_section_slide(prs_id, "Zweiter Abschnitt", "")
    print(result)

    # Add another content slide
    result = await tools.add_content_slide(prs_id, "Hauptteil", "",  [
        "Erster Punkt im Hauptteil.",
        "Noch ein wichtiger Punkt.",
        "Abschließender Punkt."
    ])
    print(result)

    # Add an image slide
    # result = await tools.add_image_slide(
    #     prs_id,
    #     "Bildbeispiel",
    #     "",
    #     "Wird neben dem Bild angezeigt.",
    #     "https://www.pcs-campus.de/wp-content/uploads/2018/03/bildformat_jpg_600_pex.jpg",
    #     "Dies ist ein Testbild."
    # )
    # print(result)

    # Add a table slide
    result = await tools.add_table_slide(
        prs_id,
        "Tabellenbeispiel",
        "Nice",
        ["Name", "Alter", "Stadt"],
        [
            ["Max", "30", "Berlin"],
            ["Anna", "25", "München"],
            ["Tom", "35", "Hamburg"]
        ]
    )
    print(result)

    # Save and print download info
    result = await tools.save_presentation_base64(prs_id)
    print(result)

    # Save the presentation as a .pptx file on disk
    file_path = f"{prs_id}.pptx"
    # save_result = tools.save_presentation_to_file(prs_id, file_path)
    # print(save_result)

if __name__ == "__main__":
    asyncio.run(main())
