import uno

def add_table_caption_with_continuation():
    # Get the current document
    ctx = uno.getComponentContext()
    smgr = ctx.ServiceManager
    desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
    doc = desktop.getCurrentComponent()

    # Ensure it's a Writer document
    if not doc.supportsService("com.sun.star.text.TextDocument"):
        raise Exception("This macro works only with Writer documents!")

    # Access the tables in the document
    tables = doc.TextTables
    if len(tables) == 0:
        raise Exception("No tables found in the document!")

    # Iterate through tables and add captions
    for table_name in tables.ElementNames:
        table = tables.getByName(table_name)
        
        # Insert the initial caption above the table
        cursor = doc.Text.createTextCursor()
        cursor.gotoRange(table.Anchor, False)
        cursor.goUp(1, False)  # Move cursor above the table
        doc.Text.insertString(cursor, f"Table {table_name}: Caption\n", False)
        
        # Add "continuation" caption if the table spans multiple pages
        text_range = table.Anchor
        table_view_cursor = doc.CurrentController.getViewCursor()
        table_view_cursor.gotoRange(text_range, False)
        
        # Check if the table spans multiple pages
        if table_view_cursor.Page > 1:
            # Move cursor to start of table on the second page
            cursor.gotoRange(table.Anchor, False)
            cursor.goDown(1, False)
            doc.Text.insertString(cursor, f"Table {table_name}: Continuation\n", False)